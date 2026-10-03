"""Spatial cross-fitted shadow audit for M1 motion uncertainty.

Diagnostic only. Four contiguous image quadrants are held out in turn. For each
fold, camera twist and raw covariance are estimated from baseline-static pixels
outside the held-out quadrant, then residual statistics are evaluated only on
the held-out quadrant. This removes direct pixel reuse between the M1 fit and
its evaluation set.

The existing LOSO covariance calibration is a sensitivity view: it was not
trained specifically for the cross-fitted estimator and must not be interpreted
as an absolutely calibrated chi-square model.

V2 additionally decomposes the projected 2x2 pose covariance into scale and
shape controls without any additional M1 fits:
  * LOSO full: calibrated scale + calibrated shape.
  * LOSO-shape/raw-scale: calibrated 2D covariance shape, raw projected trace.
  * raw-shape/LOSO-scale: raw 2D covariance shape, calibrated projected trace.
  * LOSO-scale/isotropic: calibrated projected trace, isotropic 2D shape.
These are diagnostic ablations only and never change runtime SLAM state.
"""

from pathlib import Path

import torch

from utils.m1_mapping_uncertainty import M1MappingSignal
from utils.m1_motion_uncertainty import (
    _depth_flow_jacobian,
    _interaction_matrix_per_pixel,
    _maha2,
    _quantiles,
)


def _psd(P, rel_floor=1.0e-12):
    P = 0.5 * (P.double() + P.double().T)
    eig, vec = torch.linalg.eigh(P)
    scale = eig.abs().max().clamp_min(1.0e-18)
    eig = eig.clamp_min(scale * float(rel_floor))
    out = (vec * eig.unsqueeze(0)) @ vec.T
    return 0.5 * (out + out.T)


def _quadrant_masks(height, width, device):
    """Four contiguous, non-overlapping spatial holdouts covering the image."""
    ymid = height // 2
    xmid = width // 2
    masks = []
    for y0, y1, x0, x1 in (
        (0, ymid, 0, xmid),
        (0, ymid, xmid, width),
        (ymid, height, 0, xmid),
        (ymid, height, xmid, width),
    ):
        m = torch.zeros((height, width), dtype=torch.bool, device=device)
        m[y0:y1, x0:x1] = True
        masks.append(m)
    return masks


def _auc_static_dynamic(static_scores, dynamic_scores):
    """Tie-aware Mann-Whitney AUC computed from sorted score groups."""
    s = static_scores[torch.isfinite(static_scores)].double().cpu()
    d = dynamic_scores[torch.isfinite(dynamic_scores)].double().cpu()
    n0, n1 = int(s.numel()), int(d.numel())
    if n0 == 0 or n1 == 0:
        return float("nan")
    scores = torch.cat([s, d])
    labels = torch.cat(
        [torch.zeros(n0, dtype=torch.float64), torch.ones(n1, dtype=torch.float64)]
    )
    order = torch.argsort(scores)
    xs = scores[order]
    ys = labels[order]
    _, counts = torch.unique_consecutive(xs, return_counts=True)
    group_id = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    pos = torch.zeros(counts.numel(), dtype=torch.float64)
    pos.scatter_add_(0, group_id, ys)
    neg = counts.double() - pos
    neg_before = torch.cumsum(neg, dim=0) - neg
    wins = (pos * neg_before + 0.5 * pos * neg).sum()
    return float(wins / float(n0 * n1))


def _pose_maha(residual, r00, r01, r11, pose):
    return _maha2(
        residual,
        r00 + pose[..., 0, 0],
        r01 + pose[..., 0, 1],
        r11 + pose[..., 1, 1],
    )


def _scale_cov_to_trace(cov, target_trace):
    """Preserve each 2x2 covariance shape while matching a target trace."""
    trace = (cov[..., 0, 0] + cov[..., 1, 1]).clamp_min(1.0e-20)
    factor = target_trace.clamp_min(0) / trace
    return cov * factor[..., None, None]


def _isotropic_cov_from_trace(trace):
    """2x2 isotropic PSD covariance with the requested trace."""
    out = torch.zeros((*trace.shape, 2, 2), device=trace.device, dtype=trace.dtype)
    half = 0.5 * trace.clamp_min(0)
    out[..., 0, 0] = half
    out[..., 1, 1] = half
    return out


def _group_stats(mask, scores, pose_raw_ratio, pose_cal_ratio):
    out = {
        "n": int(mask.sum().detach().cpu()),
        "pose_raw_to_observation_trace_ratio": _quantiles(pose_raw_ratio[mask]),
        "pose_loso_to_observation_trace_ratio": _quantiles(pose_cal_ratio[mask]),
    }
    for name, values in scores.items():
        out[name] = _quantiles(values[mask])
    return out


class M1MotionCrossfitAudit:
    """Four-fold quadrant cross-fit; never mutates runtime SLAM state."""

    def __init__(self, signal_file, fold, eig_floor_rel=1.0e-10):
        self.calibrator = M1MappingSignal(
            signal_file=signal_file,
            fold=fold,
            eig_floor_rel=eig_floor_rel,
        )
        self.fold = str(fold)
        self.block_size = int(self.calibrator.block_size)

    @torch.no_grad()
    def evaluate_and_save(
        self,
        *,
        frame,
        depth,
        flow_px,
        K,
        flow_var_px2,
        depth_var_m2,
        fb_valid,
        initial_static_mask,
        baseline_dynamic_mask,
        static_fit_mask,
        fit_fn,
        fit_kwargs,
        save_dir,
        dystart=None,
    ):
        H, W = depth.shape
        device, dtype = depth.device, depth.dtype
        measured = flow_px.permute(1, 2, 0)
        L = _interaction_matrix_per_pixel(depth, K)
        quadrants = _quadrant_masks(H, W, device)

        base_eval_valid = (
            torch.isfinite(depth)
            & (depth > 0)
            & torch.isfinite(measured).all(dim=-1)
            & torch.isfinite(flow_var_px2)
            & (flow_var_px2 > 0)
            & torch.isfinite(depth_var_m2)
            & (depth_var_m2 >= 0)
            & fb_valid.bool()
            & initial_static_mask.bool()
        )

        score_names = (
            "raw_resid_px",
            "d2_observation_only",
            "d2_predictive_raw",
            "d2_predictive_loso_sensitivity",
            "d2_ablate_loso_shape_raw_scale",
            "d2_ablate_raw_shape_loso_scale",
            "d2_ablate_loso_scale_isotropic",
        )
        folds = []
        pooled = {
            "static": {name: [] for name in score_names},
            "dynamic": {name: [] for name in score_names},
        }
        for name in ("static", "dynamic"):
            pooled[name]["ratio_raw"] = []
            pooled[name]["ratio_cal"] = []

        for fold_idx, holdout in enumerate(quadrants):
            train_mask = static_fit_mask.bool() & (~holdout)
            result = fit_fn(
                depth,
                flow_px,
                K,
                train_mask,
                flow_var_px2,
                depth_var_m2,
                fixed_xi=None,
                **fit_kwargs,
            )
            xi = result["xi"].detach().to(device=device, dtype=dtype)
            P_raw = result["cov_clusters"].get(self.block_size)
            if P_raw is None:
                raise KeyError(
                    f"Cross-fit M1 result has no cluster covariance for block {self.block_size}"
                )
            P_raw = _psd(P_raw).to(device=device, dtype=dtype)
            P_cal = self.calibrator.calibrated_covariance(P_raw.detach().cpu())
            P_cal = _psd(P_cal).to(device=device, dtype=dtype)

            pred = torch.einsum("hwai,i->hwa", L, xi)
            residual = measured - pred
            residual_norm = torch.linalg.vector_norm(residual, dim=-1)

            J_D = _depth_flow_jacobian(depth, K, xi)
            ju, jv = J_D[..., 0], J_D[..., 1]
            r00 = flow_var_px2 + depth_var_m2 * ju * ju + 1.0e-8
            r11 = flow_var_px2 + depth_var_m2 * jv * jv + 1.0e-8
            r01 = depth_var_m2 * ju * jv
            trace_obs = (r00 + r11).clamp_min(1.0e-12)

            pose_raw = torch.einsum("hwai,ij,hwbj->hwab", L, P_raw, L)
            pose_cal = torch.einsum("hwai,ij,hwbj->hwab", L, P_cal, L)
            trace_raw = (pose_raw[..., 0, 0] + pose_raw[..., 1, 1]).clamp_min(0)
            trace_cal = (pose_cal[..., 0, 0] + pose_cal[..., 1, 1]).clamp_min(0)

            # Scale/shape decomposition in projected 2D flow-covariance space.
            # These controls require no extra probabilistic fits.
            pose_loso_shape_raw_scale = _scale_cov_to_trace(pose_cal, trace_raw)
            pose_raw_shape_loso_scale = _scale_cov_to_trace(pose_raw, trace_cal)
            pose_loso_scale_isotropic = _isotropic_cov_from_trace(trace_cal)

            d2_obs = _maha2(residual, r00, r01, r11)
            scores = {
                "raw_resid_px": residual_norm,
                "d2_observation_only": d2_obs,
                "d2_predictive_raw": _pose_maha(residual, r00, r01, r11, pose_raw),
                "d2_predictive_loso_sensitivity": _pose_maha(residual, r00, r01, r11, pose_cal),
                "d2_ablate_loso_shape_raw_scale": _pose_maha(
                    residual, r00, r01, r11, pose_loso_shape_raw_scale
                ),
                "d2_ablate_raw_shape_loso_scale": _pose_maha(
                    residual, r00, r01, r11, pose_raw_shape_loso_scale
                ),
                "d2_ablate_loso_scale_isotropic": _pose_maha(
                    residual, r00, r01, r11, pose_loso_scale_isotropic
                ),
            }
            ratio_raw = trace_raw / trace_obs
            ratio_cal = trace_cal / trace_obs

            eval_mask = base_eval_valid & holdout
            dyn = baseline_dynamic_mask.bool() & eval_mask
            sta = (~baseline_dynamic_mask.bool()) & eval_mask

            invariant_keys = {
                "raw_pred_gt_obs_fraction": "d2_predictive_raw",
                "loso_pred_gt_obs_fraction": "d2_predictive_loso_sensitivity",
                "loso_shape_raw_scale_pred_gt_obs_fraction": "d2_ablate_loso_shape_raw_scale",
                "raw_shape_loso_scale_pred_gt_obs_fraction": "d2_ablate_raw_shape_loso_scale",
                "loso_scale_isotropic_pred_gt_obs_fraction": "d2_ablate_loso_scale_isotropic",
            }
            invariants = {}
            for key, score_name in invariant_keys.items():
                invariants[key] = (
                    float(
                        (
                            scores[score_name]
                            > d2_obs * (1.0 + 1.0e-5) + 1.0e-6
                        )[eval_mask]
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    )
                    if bool(eval_mask.any())
                    else float("nan")
                )

            # Verify the intended trace matching numerically on the held-out region.
            def trace_relerr(cov, target):
                tr = cov[..., 0, 0] + cov[..., 1, 1]
                den = target.abs().clamp_min(1.0e-12)
                return float(((tr - target).abs() / den)[eval_mask].max().detach().cpu()) \
                    if bool(eval_mask.any()) else float("nan")

            invariants.update({
                "loso_shape_raw_scale_trace_relerr_max": trace_relerr(
                    pose_loso_shape_raw_scale, trace_raw
                ),
                "raw_shape_loso_scale_trace_relerr_max": trace_relerr(
                    pose_raw_shape_loso_scale, trace_cal
                ),
                "loso_scale_isotropic_trace_relerr_max": trace_relerr(
                    pose_loso_scale_isotropic, trace_cal
                ),
            })

            folds.append({
                "fold": int(fold_idx),
                "n_train_static": int(train_mask.sum().detach().cpu()),
                "n_eval": int(eval_mask.sum().detach().cpu()),
                "n_static": int(sta.sum().detach().cpu()),
                "n_dynamic": int(dyn.sum().detach().cpu()),
                "xi": xi.detach().cpu(),
                "covariance_mode": str(result.get("covariance_mode")),
                "cluster_count": int(result.get("cluster_counts", {}).get(self.block_size, 0)),
                "groups": {
                    "baseline_static": _group_stats(sta, scores, ratio_raw, ratio_cal),
                    "baseline_dynamic": _group_stats(dyn, scores, ratio_raw, ratio_cal),
                },
                "invariants": invariants,
            })

            for name, mask in (("static", sta), ("dynamic", dyn)):
                for score_name in score_names:
                    pooled[name][score_name].append(scores[score_name][mask].detach().cpu())
                pooled[name]["ratio_raw"].append(ratio_raw[mask].detach().cpu())
                pooled[name]["ratio_cal"].append(ratio_cal[mask].detach().cpu())

        def cat(name, key):
            xs = pooled[name][key]
            return torch.cat(xs) if xs else torch.empty(0)

        pooled_groups = {}
        for name in ("static", "dynamic"):
            pooled_groups["baseline_" + name] = {
                "n": int(cat(name, "d2_observation_only").numel()),
                **{score_name: _quantiles(cat(name, score_name)) for score_name in score_names},
                "pose_raw_to_observation_trace_ratio": _quantiles(cat(name, "ratio_raw")),
                "pose_loso_to_observation_trace_ratio": _quantiles(cat(name, "ratio_cal")),
            }

        auc = {
            score_name: _auc_static_dynamic(
                cat("static", score_name), cat("dynamic", score_name)
            )
            for score_name in score_names
        }

        payload = {
            "method": "m1_motion_spatial_crossfit_shadow_v2_scale_shape",
            "frame": int(frame),
            "dystart": None if dystart is None else int(dystart),
            "fold": self.fold,
            "crossfit": {
                "scheme": "four_contiguous_quadrants",
                "n_folds": 4,
                "training_uses_baseline_static_only": True,
                "evaluation_is_held_out_spatial_region": True,
                "loso_calibration_is_sensitivity_only": True,
                "ablation_space": "projected_2d_flow_covariance",
                "ablation_requires_extra_fits": False,
            },
            "block_size": self.block_size,
            "folds": folds,
            "pooled_groups": pooled_groups,
            "auc_dynamic_vs_static": auc,
        }

        out_dir = Path(save_dir) / "m1_motion_crossfit"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_dir / f"{int(frame):06d}.pt")
        return payload
