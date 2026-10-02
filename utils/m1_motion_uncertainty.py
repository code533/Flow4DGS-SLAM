"""Shadow diagnostics for propagating M1 camera-motion uncertainty to flow residuals.

This module is deliberately diagnostic-only. It never mutates a camera pose,
motion mask, Gaussian state, loss, keyframe decision, or optimizer state.

For each pixel p it evaluates

    r_p = f_p - L_p xi
    R_p = sigma_F,p^2 I + sigma_D,p^2 J_D,p J_D,p^T
    S_p = R_p + L_p P_xi L_p^T

and records both the observation-only innovation r^T R^-1 r and the
pose-predictive innovation r^T S^-1 r. P_xi is the held-out/LOSO calibrated
M1 cluster covariance loaded through M1MappingSignal.
"""

import math
from pathlib import Path

import torch

from utils.m1_mapping_uncertainty import M1MappingSignal


CHI2_DF2_Q95 = -2.0 * math.log(0.05)
CHI2_DF2_Q99 = -2.0 * math.log(0.01)


def _mesh(height, width, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return x, y


def _interaction_matrix_per_pixel(depth, K):
    """Return L with shape [H,W,2,6], matching Flow4DGS' twist convention."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y = _mesh(H, W, depth.device, depth.dtype)
    Z = depth.clamp_min(1.0e-6)

    du_dt = torch.stack(
        [-fx / Z, torch.zeros_like(Z), (x - cx) / Z], dim=-1
    )
    dv_dt = torch.stack(
        [torch.zeros_like(Z), -fy / Z, (y - cy) / Z], dim=-1
    )
    du_dw = torch.stack(
        [
            (x - cx) * (y - cy) / fy,
            -(fx + (x - cx) * (x - cx) / fx),
            (y - cy),
        ],
        dim=-1,
    )
    dv_dw = torch.stack(
        [
            (fy + (y - cy) * (y - cy) / fy),
            -(x - cx) * (y - cy) / fx,
            -(x - cx),
        ],
        dim=-1,
    )
    return torch.stack(
        [torch.cat([du_dt, du_dw], dim=-1), torch.cat([dv_dt, dv_dw], dim=-1)],
        dim=2,
    )


def _depth_flow_jacobian(depth, K, xi):
    """Return d(flow_u, flow_v)/dZ with shape [H,W,2]."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y = _mesh(H, W, depth.device, depth.dtype)
    u = x - cx
    v = y - cy
    Z = depth.clamp_min(1.0e-6)
    tx, ty, tz = xi[0], xi[1], xi[2]
    dFu_dZ = (fx * tx - u * tz) / (Z * Z)
    dFv_dZ = (fy * ty - v * tz) / (Z * Z)
    return torch.stack([dFu_dZ, dFv_dZ], dim=-1)


def _maha2(residual, c00, c01, c11, eps=1.0e-12):
    """Mahalanobis square for a symmetric 2x2 covariance field."""
    det = (c00 * c11 - c01 * c01).clamp_min(eps)
    ru = residual[..., 0]
    rv = residual[..., 1]
    return (
        c11 * ru * ru - 2.0 * c01 * ru * rv + c00 * rv * rv
    ) / det


def _quantiles(values):
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            "q50": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
        }
    q = torch.quantile(
        values.float(),
        torch.tensor([0.50, 0.90, 0.95, 0.99], device=values.device),
    ).detach().cpu().tolist()
    return {
        "q50": float(q[0]),
        "q90": float(q[1]),
        "q95": float(q[2]),
        "q99": float(q[3]),
    }


def _fraction(mask):
    return float(mask.float().mean().detach().cpu()) if mask.numel() else float("nan")


def _hist_log10(values, bins, lo=-4.0, hi=4.0):
    values = values[torch.isfinite(values) & (values >= 0)]
    if values.numel() == 0:
        return {
            "lo": lo,
            "hi": hi,
            "bins": int(bins),
            "counts": [0] * int(bins),
            "underflow": 0,
            "overflow": 0,
        }
    logv_raw = torch.log10(values.clamp_min(1.0e-30))
    underflow = int((logv_raw < lo).sum().detach().cpu())
    overflow = int((logv_raw > hi).sum().detach().cpu())
    clipped = logv_raw.clamp(min=lo, max=hi)
    counts = torch.histc(clipped.float(), bins=int(bins), min=lo, max=hi)
    return {
        "lo": float(lo),
        "hi": float(hi),
        "bins": int(bins),
        "counts": [int(v) for v in counts.detach().cpu().tolist()],
        "underflow": underflow,
        "overflow": overflow,
    }


class M1MotionShadowAudit:
    """LOSO-calibrated, behavior-preserving motion-segmentation audit."""

    def __init__(self, signal_file, fold, eig_floor_rel=1.0e-10, hist_bins=64):
        self.calibrator = M1MappingSignal(
            signal_file=signal_file,
            fold=fold,
            eig_floor_rel=eig_floor_rel,
        )
        self.fold = str(fold)
        self.block_size = int(self.calibrator.block_size)
        self.hist_bins = int(hist_bins)
        if self.hist_bins < 8:
            raise ValueError("m1_motion_hist_bins must be >= 8")

    @torch.no_grad()
    def evaluate_and_save(
        self,
        *,
        frame,
        depth,
        flow_px,
        K,
        xi,
        P_raw,
        flow_var_px2,
        depth_var_m2,
        fb_valid,
        initial_static_mask,
        baseline_dynamic_mask,
        save_dir,
        covariance_mode,
        dystart=None,
    ):
        H, W = depth.shape
        if tuple(flow_px.shape) != (2, H, W):
            raise ValueError(f"Expected flow [2,{H},{W}], got {tuple(flow_px.shape)}")

        device, dtype = depth.device, depth.dtype
        P_cal = self.calibrator.calibrated_covariance(P_raw.detach().cpu())
        P_cal = P_cal.to(device=device, dtype=dtype)
        P_cal = 0.5 * (P_cal + P_cal.T)
        peig, pvec = torch.linalg.eigh(P_cal.double())
        pmax = peig.abs().max().clamp_min(1.0e-18)
        peig_psd = peig.clamp_min(pmax * 1.0e-12)
        P_cal = ((pvec * peig_psd.unsqueeze(0)) @ pvec.T).to(dtype)
        P_cal = 0.5 * (P_cal + P_cal.T)

        L = _interaction_matrix_per_pixel(depth, K)
        pred = torch.einsum("hwai,i->hwa", L, xi)
        measured = flow_px.permute(1, 2, 0)
        residual = measured - pred
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)

        J_D = _depth_flow_jacobian(depth, K, xi)
        ju, jv = J_D[..., 0], J_D[..., 1]
        r00 = flow_var_px2 + depth_var_m2 * ju * ju + 1.0e-8
        r11 = flow_var_px2 + depth_var_m2 * jv * jv + 1.0e-8
        r01 = depth_var_m2 * ju * jv

        pose_cov = torch.einsum("hwai,ij,hwbj->hwab", L, P_cal, L)
        s00 = r00 + pose_cov[..., 0, 0]
        s11 = r11 + pose_cov[..., 1, 1]
        s01 = r01 + pose_cov[..., 0, 1]

        d2_obs = _maha2(residual, r00, r01, r11)
        d2_pred = _maha2(residual, s00, s01, s11)

        trace_obs = (r00 + r11).clamp_min(1.0e-12)
        trace_pose = (
            pose_cov[..., 0, 0] + pose_cov[..., 1, 1]
        ).clamp_min(0.0)
        pose_to_obs = trace_pose / trace_obs
        pose_fraction = trace_pose / (trace_pose + trace_obs)

        valid = (
            torch.isfinite(depth)
            & (depth > 0)
            & torch.isfinite(measured).all(dim=-1)
            & torch.isfinite(flow_var_px2)
            & (flow_var_px2 > 0)
            & torch.isfinite(depth_var_m2)
            & (depth_var_m2 >= 0)
            & fb_valid.bool()
            & initial_static_mask.bool()
            & torch.isfinite(d2_obs)
            & torch.isfinite(d2_pred)
        )
        baseline_dynamic = baseline_dynamic_mask.bool() & valid
        baseline_static = (~baseline_dynamic_mask.bool()) & valid

        n_valid = int(valid.sum().detach().cpu())
        n_static = int(baseline_static.sum().detach().cpu())
        n_dynamic = int(baseline_dynamic.sum().detach().cpu())

        def group_stats(mask):
            return {
                "n": int(mask.sum().detach().cpu()),
                "raw_resid_px": _quantiles(residual_norm[mask]),
                "d2_observation_only": _quantiles(d2_obs[mask]),
                "d2_pose_predictive": _quantiles(d2_pred[mask]),
                "pose_to_observation_trace_ratio": _quantiles(pose_to_obs[mask]),
                "pose_fraction_of_predictive_trace": _quantiles(pose_fraction[mask]),
            }

        thresholds = {"q95": CHI2_DF2_Q95, "q99": CHI2_DF2_Q99}
        disagreement = {}
        for name, threshold in thresholds.items():
            obs_dynamic = (d2_obs > threshold) & valid
            pred_dynamic = (d2_pred > threshold) & valid
            disagreement[name] = {
                "threshold": float(threshold),
                "baseline_vs_predictive_disagree_fraction": _fraction(
                    (baseline_dynamic ^ pred_dynamic)[valid]
                ),
                "baseline_dynamic_to_predictive_static_fraction": _fraction(
                    (~pred_dynamic)[baseline_dynamic]
                ),
                "baseline_static_to_predictive_dynamic_fraction": _fraction(
                    pred_dynamic[baseline_static]
                ),
                "observation_dynamic_to_predictive_static_fraction": _fraction(
                    (~pred_dynamic)[obs_dynamic]
                ),
                "predictive_dynamic_but_observation_static_fraction": _fraction(
                    pred_dynamic[(~obs_dynamic) & valid]
                ),
                "observation_dynamic_fraction": _fraction(obs_dynamic[valid]),
                "predictive_dynamic_fraction": _fraction(pred_dynamic[valid]),
            }

        payload = {
            "method": "m1_probabilistic_motion_shadow_v1",
            "frame": int(frame),
            "dystart": None if dystart is None else int(dystart),
            "fold": self.fold,
            "block_size": self.block_size,
            "covariance_mode": str(covariance_mode),
            "n_valid": n_valid,
            "n_baseline_static": n_static,
            "n_baseline_dynamic": n_dynamic,
            "baseline_dynamic_fraction": (
                float(n_dynamic) / n_valid if n_valid else float("nan")
            ),
            "P_xi_calibrated": P_cal.detach().cpu(),
            "P_xi_calibrated_min_eig_before_floor": float(
                peig.min().detach().cpu()
            ),
            "chi2_df2_reference": thresholds,
            "reference_thresholds_are_diagnostic_only": True,
            "groups": {
                "all": group_stats(valid),
                "baseline_static": group_stats(baseline_static),
                "baseline_dynamic": group_stats(baseline_dynamic),
            },
            "disagreement": disagreement,
            "hist_log10_d2": {
                "baseline_static_observation_only": _hist_log10(
                    d2_obs[baseline_static], self.hist_bins
                ),
                "baseline_static_pose_predictive": _hist_log10(
                    d2_pred[baseline_static], self.hist_bins
                ),
                "baseline_dynamic_observation_only": _hist_log10(
                    d2_obs[baseline_dynamic], self.hist_bins
                ),
                "baseline_dynamic_pose_predictive": _hist_log10(
                    d2_pred[baseline_dynamic], self.hist_bins
                ),
            },
            "invariants": {
                "predictive_d2_gt_observation_d2_fraction": _fraction(
                    (d2_pred > d2_obs * (1.0 + 1.0e-5) + 1.0e-6)[valid]
                )
            },
        }

        out_dir = Path(save_dir) / "m1_motion_shadow"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{int(frame):06d}.pt"
        torch.save(payload, out_path)
        return payload
