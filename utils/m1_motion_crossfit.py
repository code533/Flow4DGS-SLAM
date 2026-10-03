"""Spatial cross-fitted shadow audit for M1 motion uncertainty.

Diagnostic only. Four contiguous image quadrants are held out in turn. For each
fold, camera twist and raw covariance are estimated from baseline-static pixels
outside the held-out quadrant, then residual statistics are evaluated only on
the held-out quadrant. This removes direct pixel reuse between the M1 fit and
its evaluation set.

The existing LOSO covariance calibration is also reported as a sensitivity
view, but it was not trained specifically for the cross-fitted estimator and
must not be interpreted as an absolutely calibrated chi-square model.
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
    labels = torch.cat([torch.zeros(n0, dtype=torch.float64), torch.ones(n1, dtype=torch.float64)])
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


def _group_stats(mask, residual_norm, d2_obs, d2_raw, d2_cal, pose_raw_ratio, pose_cal_ratio):
    return {
        "n": int(mask.sum().detach().cpu()),
        "raw_resid_px": _quantiles(residual_norm[mask]),
        "d2_observation_only": _quantiles(d2_obs[mask]),
        "d2_predictive_raw": _quantiles(d2_raw[mask]),
        "d2_predictive_loso_sensitivity": _quantiles(d2_cal[mask]),
        "pose_raw_to_observation_trace_ratio": _quantiles(pose_raw_ratio[mask]),
        "pose_loso_to_observation_trace_ratio": _quantiles(pose_cal_ratio[mask]),
    }


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

        folds = []
        pooled = {
            "static": {"raw": [], "obs": [], "pred_raw": [], "pred_cal": [], "ratio_raw": [], "ratio_cal": []},
            "dynamic": {"raw": [], "obs": [], "pred_raw": [], "pred_cal": [], "ratio_raw": [], "ratio_cal": []},
        }

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

            d2_obs = _maha2(residual, r00, r01, r11)
            d2_raw = _maha2(
                residual,
                r00 + pose_raw[..., 0, 0],
                r01 + pose_raw[..., 0, 1],
                r11 + pose_raw[..., 1, 1],
            )
            d2_cal = _maha2(
                residual,
                r00 + pose_cal[..., 0, 0],
                r01 + pose_cal[..., 0, 1],
                r11 + pose_cal[..., 1, 1],
            )
            ratio_raw = (pose_raw[..., 0, 0] + pose_raw[..., 1, 1]).clamp_min(0) / trace_obs
            ratio_cal = (pose_cal[..., 0, 0] + pose_cal[..., 1, 1]).clamp_min(0) / trace_obs

            eval_mask = base_eval_valid & holdout
            dyn = baseline_dynamic_mask.bool() & eval_mask
            sta = (~baseline_dynamic_mask.bool()) & eval_mask

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
                    "baseline_static": _group_stats(sta, residual_norm, d2_obs, d2_raw, d2_cal, ratio_raw, ratio_cal),
                    "baseline_dynamic": _group_stats(dyn, residual_norm, d2_obs, d2_raw, d2_cal, ratio_raw, ratio_cal),
                },
                "invariants": {
                    "raw_pred_gt_obs_fraction": float(
                        ((d2_raw > d2_obs * (1.0 + 1.0e-5) + 1.0e-6)[eval_mask]).float().mean().detach().cpu()
                    ) if bool(eval_mask.any()) else float("nan"),
                    "loso_pred_gt_obs_fraction": float(
                        ((d2_cal > d2_obs * (1.0 + 1.0e-5) + 1.0e-6)[eval_mask]).float().mean().detach().cpu()
                    ) if bool(eval_mask.any()) else float("nan"),
                },
            })

            for name, mask in (("static", sta), ("dynamic", dyn)):
                pooled[name]["raw"].append(residual_norm[mask].detach().cpu())
                pooled[name]["obs"].append(d2_obs[mask].detach().cpu())
                pooled[name]["pred_raw"].append(d2_raw[mask].detach().cpu())
                pooled[name]["pred_cal"].append(d2_cal[mask].detach().cpu())
                pooled[name]["ratio_raw"].append(ratio_raw[mask].detach().cpu())
                pooled[name]["ratio_cal"].append(ratio_cal[mask].detach().cpu())

        def cat(name, key):
            xs = pooled[name][key]
            return torch.cat(xs) if xs else torch.empty(0)

        pooled_groups = {}
        for name in ("static", "dynamic"):
            vals = {k: cat(name, k) for k in pooled[name]}
            pooled_groups["baseline_" + name] = {
                "n": int(vals["obs"].numel()),
                "raw_resid_px": _quantiles(vals["raw"]),
                "d2_observation_only": _quantiles(vals["obs"]),
                "d2_predictive_raw": _quantiles(vals["pred_raw"]),
                "d2_predictive_loso_sensitivity": _quantiles(vals["pred_cal"]),
                "pose_raw_to_observation_trace_ratio": _quantiles(vals["ratio_raw"]),
                "pose_loso_to_observation_trace_ratio": _quantiles(vals["ratio_cal"]),
            }

        static_vals = {k: cat("static", k) for k in pooled["static"]}
        dynamic_vals = {k: cat("dynamic", k) for k in pooled["dynamic"]}
        auc = {
            "raw_resid_px": _auc_static_dynamic(static_vals["raw"], dynamic_vals["raw"]),
            "d2_observation_only": _auc_static_dynamic(static_vals["obs"], dynamic_vals["obs"]),
            "d2_predictive_raw": _auc_static_dynamic(static_vals["pred_raw"], dynamic_vals["pred_raw"]),
            "d2_predictive_loso_sensitivity": _auc_static_dynamic(static_vals["pred_cal"], dynamic_vals["pred_cal"]),
        }

        payload = {
            "method": "m1_motion_spatial_crossfit_shadow_v1",
            "frame": int(frame),
            "dystart": None if dystart is None else int(dystart),
            "fold": self.fold,
            "crossfit": {
                "scheme": "four_contiguous_quadrants",
                "n_folds": 4,
                "training_uses_baseline_static_only": True,
                "evaluation_is_held_out_spatial_region": True,
                "loso_calibration_is_sensitivity_only": True,
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
