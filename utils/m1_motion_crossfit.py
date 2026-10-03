"""Spatial cross-fitted shadow audit for M1 motion uncertainty.

Diagnostic only. Four contiguous image quadrants are held out in turn. For each
fold, camera twist and raw covariance are estimated from baseline-static pixels
outside the held-out quadrant, then residual statistics are evaluated only on
the held-out quadrant. This removes direct pixel reuse between the M1 fit and
its evaluation set.

The existing LOSO covariance calibration is a sensitivity view: it was not
trained specifically for the cross-fitted estimator and must not be interpreted
as an absolutely calibrated chi-square model.

V2 decomposes projected 2x2 pose covariance into scale and shape controls.
V3 adds scale-source controls, all computed from the same cross-fit:
  * full LOSO projected scale with isotropic 2D shape (existing V2 control);
  * geometry-only scale from tr(L L^T), median-matched per held-out fold;
  * constant scale equal to the held-out-fold median LOSO projected trace;
  * deterministically shuffled LOSO projected scale within the held-out fold.

V3 also records scalar associations for LOSO projected scale, interaction-matrix
geometry, depth, and normalized image radius. No source control uses dynamic
labels to set its scale, and the deterministic shuffle uses a local CPU RNG so
it does not perturb the SLAM process RNG state.
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


def _median_match(values, target, mask):
    """Scale values so their masked median matches target's masked median."""
    if not bool(mask.any()):
        return values, float("nan")
    src_med = torch.median(values[mask]).clamp_min(1.0e-20)
    tgt_med = torch.median(target[mask]).clamp_min(0)
    factor = tgt_med / src_med
    return values * factor, float(factor.detach().cpu())


def _shuffle_on_mask(values, mask, seed):
    """Exact masked-value permutation using a local CPU RNG only."""
    out = values.clone()
    flat_mask = mask.reshape(-1)
    idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
    n = int(idx.numel())
    if n <= 1:
        return out
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen).to(device=idx.device)
    out_flat = out.reshape(-1)
    src = values.reshape(-1)[idx]
    out_flat[idx] = src[perm]
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

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        fx = K[0, 0].to(device=device, dtype=dtype)
        fy = K[1, 1].to(device=device, dtype=dtype)
        cx = K[0, 2].to(device=device, dtype=dtype)
        cy = K[1, 2].to(device=device, dtype=dtype)
        radius2 = ((xx - cx) / fx.clamp_min(1.0e-12)) ** 2 + (
            (yy - cy) / fy.clamp_min(1.0e-12)
        ) ** 2
        trace_LL = (L * L).sum(dim=(-2, -1)).clamp_min(0)

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

        base_score_names = (
            "raw_resid_px",
            "d2_observation_only",
            "d2_predictive_raw",
            "d2_predictive_loso_sensitivity",
            "d2_ablate_loso_shape_raw_scale",
            "d2_ablate_raw_shape_loso_scale",
            "d2_ablate_loso_scale_isotropic",
        )
        source_d2_names = (
            "d2_source_geometry_only_matched",
            "d2_source_constant_scale",
            "d2_source_shuffled_scale",
        )
        source_scalar_names = (
            "source_scale_loso_trace",
            "source_geometry_trace_LL",
            "source_depth_m",
            "source_radius2_norm",
            "source_scale_over_trace_LL",
        )
        score_names = base_score_names + source_d2_names + source_scalar_names

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

            eval_mask = base_eval_valid & holdout
            dyn = baseline_dynamic_mask.bool() & eval_mask
            sta = (~baseline_dynamic_mask.bool()) & eval_mask

            # V2 projected 2D scale/shape decomposition.
            pose_loso_shape_raw_scale = _scale_cov_to_trace(pose_cal, trace_raw)
            pose_raw_shape_loso_scale = _scale_cov_to_trace(pose_raw, trace_cal)
            pose_loso_scale_isotropic = _isotropic_cov_from_trace(trace_cal)

            # V3 scale-source controls. None uses dynamic/static labels.
            # Geometry-only: use tr(L I L^T)=||L||_F^2, then match only the
            # held-out-fold median scale to the full LOSO projected trace.
            trace_geometry_matched, geometry_match_factor = _median_match(
                trace_LL, trace_cal, eval_mask
            )
            pose_geometry_only = _isotropic_cov_from_trace(trace_geometry_matched)

            if bool(eval_mask.any()):
                full_trace_median = torch.median(trace_cal[eval_mask]).clamp_min(0)
            else:
                full_trace_median = torch.zeros((), device=device, dtype=dtype)
            trace_constant = torch.ones_like(trace_cal) * full_trace_median
            pose_constant = _isotropic_cov_from_trace(trace_constant)

            shuffle_seed = (
                (int(frame) + 1) * 1000003 + (int(fold_idx) + 1) * 9176
            ) % 2147483647
            trace_shuffled = _shuffle_on_mask(trace_cal, eval_mask, shuffle_seed)
            pose_shuffled = _isotropic_cov_from_trace(trace_shuffled)

            d2_obs = _maha2(residual, r00, r01, r11)
            scores = {
                "raw_resid_px": residual_norm,
                "d2_observation_only": d2_obs,
                "d2_predictive_raw": _pose_maha(residual, r00, r01, r11, pose_raw),
                "d2_predictive_loso_sensitivity": _pose_maha(
                    residual, r00, r01, r11, pose_cal
                ),
                "d2_ablate_loso_shape_raw_scale": _pose_maha(
                    residual, r00, r01, r11, pose_loso_shape_raw_scale
                ),
                "d2_ablate_raw_shape_loso_scale": _pose_maha(
                    residual, r00, r01, r11, pose_raw_shape_loso_scale
                ),
                "d2_ablate_loso_scale_isotropic": _pose_maha(
                    residual, r00, r01, r11, pose_loso_scale_isotropic
                ),
                "d2_source_geometry_only_matched": _pose_maha(
                    residual, r00, r01, r11, pose_geometry_only
                ),
                "d2_source_constant_scale": _pose_maha(
                    residual, r00, r01, r11, pose_constant
                ),
                "d2_source_shuffled_scale": _pose_maha(
                    residual, r00, r01, r11, pose_shuffled
                ),
                "source_scale_loso_trace": trace_cal,
                "source_geometry_trace_LL": trace_LL,
                "source_depth_m": depth,
                "source_radius2_norm": radius2,
                "source_scale_over_trace_LL": trace_cal / trace_LL.clamp_min(1.0e-12),
            }
            ratio_raw = trace_raw / trace_obs
            ratio_cal = trace_cal / trace_obs

            invariant_keys = {
                "raw_pred_gt_obs_fraction": "d2_predictive_raw",
                "loso_pred_gt_obs_fraction": "d2_predictive_loso_sensitivity",
                "loso_shape_raw_scale_pred_gt_obs_fraction": "d2_ablate_loso_shape_raw_scale",
                "raw_shape_loso_scale_pred_gt_obs_fraction": "d2_ablate_raw_shape_loso_scale",
                "loso_scale_isotropic_pred_gt_obs_fraction": "d2_ablate_loso_scale_isotropic",
                "source_geometry_pred_gt_obs_fraction": "d2_source_geometry_only_matched",
                "source_constant_pred_gt_obs_fraction": "d2_source_constant_scale",
                "source_shuffled_pred_gt_obs_fraction": "d2_source_shuffled_scale",
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

            def trace_relerr(cov, target):
                tr = cov[..., 0, 0] + cov[..., 1, 1]
                den = target.abs().clamp_min(1.0e-12)
                return (
                    float(
                        ((tr - target).abs() / den)[eval_mask]
                        .max()
                        .detach()
                        .cpu()
                    )
                    if bool(eval_mask.any())
                    else float("nan")
                )

            invariants.update(
                {
                    "loso_shape_raw_scale_trace_relerr_max": trace_relerr(
                        pose_loso_shape_raw_scale, trace_raw
                    ),
                    "raw_shape_loso_scale_trace_relerr_max": trace_relerr(
                        pose_raw_shape_loso_scale, trace_cal
                    ),
                    "loso_scale_isotropic_trace_relerr_max": trace_relerr(
                        pose_loso_scale_isotropic, trace_cal
                    ),
                    "source_geometry_median_match_ratio": (
                        float(
                            (
                                torch.median(trace_geometry_matched[eval_mask])
                                / torch.median(trace_cal[eval_mask]).clamp_min(1.0e-20)
                            )
                            .detach()
                            .cpu()
                        )
                        if bool(eval_mask.any())
                        else float("nan")
                    ),
                    "source_constant_median_match_ratio": (
                        float(
                            (
                                torch.median(trace_constant[eval_mask])
                                / torch.median(trace_cal[eval_mask]).clamp_min(1.0e-20)
                            )
                            .detach()
                            .cpu()
                        )
                        if bool(eval_mask.any())
                        else float("nan")
                    ),
                    "source_shuffle_mean_relerr": (
                        float(
                            (
                                (
                                    trace_shuffled[eval_mask].mean()
                                    - trace_cal[eval_mask].mean()
                                ).abs()
                                / trace_cal[eval_mask].mean().abs().clamp_min(1.0e-20)
                            )
                            .detach()
                            .cpu()
                        )
                        if bool(eval_mask.any())
                        else float("nan")
                    ),
                }
            )

            folds.append(
                {
                    "fold": int(fold_idx),
                    "n_train_static": int(train_mask.sum().detach().cpu()),
                    "n_eval": int(eval_mask.sum().detach().cpu()),
                    "n_static": int(sta.sum().detach().cpu()),
                    "n_dynamic": int(dyn.sum().detach().cpu()),
                    "xi": xi.detach().cpu(),
                    "covariance_mode": str(result.get("covariance_mode")),
                    "cluster_count": int(
                        result.get("cluster_counts", {}).get(self.block_size, 0)
                    ),
                    "source_control": {
                        "geometry_match_factor": geometry_match_factor,
                        "full_trace_median": float(full_trace_median.detach().cpu()),
                        "shuffle_seed": int(shuffle_seed),
                    },
                    "groups": {
                        "baseline_static": _group_stats(
                            sta, scores, ratio_raw, ratio_cal
                        ),
                        "baseline_dynamic": _group_stats(
                            dyn, scores, ratio_raw, ratio_cal
                        ),
                    },
                    "invariants": invariants,
                }
            )

            for name, mask in (("static", sta), ("dynamic", dyn)):
                for score_name in score_names:
                    pooled[name][score_name].append(
                        scores[score_name][mask].detach().cpu()
                    )
                pooled[name]["ratio_raw"].append(ratio_raw[mask].detach().cpu())
                pooled[name]["ratio_cal"].append(ratio_cal[mask].detach().cpu())

        def cat(name, key):
            xs = pooled[name][key]
            return torch.cat(xs) if xs else torch.empty(0)

        pooled_groups = {}
        for name in ("static", "dynamic"):
            pooled_groups["baseline_" + name] = {
                "n": int(cat(name, "d2_observation_only").numel()),
                **{
                    score_name: _quantiles(cat(name, score_name))
                    for score_name in score_names
                },
                "pose_raw_to_observation_trace_ratio": _quantiles(
                    cat(name, "ratio_raw")
                ),
                "pose_loso_to_observation_trace_ratio": _quantiles(
                    cat(name, "ratio_cal")
                ),
            }

        auc = {
            score_name: _auc_static_dynamic(
                cat("static", score_name), cat("dynamic", score_name)
            )
            for score_name in score_names
        }

        payload = {
            "method": "m1_motion_spatial_crossfit_shadow_v3_scale_source",
            "frame": int(frame),
            "dystart": None if dystart is None else int(dystart),
            "fold": self.fold,
            "crossfit": {
                "scheme": "four_contiguous_quadrants",
                "n_folds": 4,
                "training_uses_baseline_static_only": True,
                "evaluation_is_held_out_spatial_region": True,
                "loso_calibration_is_sensitivity_only": True,
                "ablation_space": "projected_flow_covariance_and_scale_source",
                "ablation_requires_extra_fits": False,
                "source_controls_use_dynamic_labels": False,
                "shuffle_uses_global_rng": False,
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
