#!/usr/bin/env python3
"""Aggregate spatial cross-fitted M1 motion-shadow diagnostics.

No intervention threshold is selected. LOSO-calibrated covariance is reported
only as a sensitivity analysis because the calibration was not trained for the
cross-fitted estimator.

V2 reports projected 2D covariance scale/shape ablations.
V3 adds scale-source controls from the same cross-fit, with no extra M1 fits:
geometry-only matched scale, fold-constant scale, and deterministically shuffled
pixel scale. It also reports scalar association with LOSO projected scale,
interaction-matrix geometry, depth, and normalized image radius.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


BASE_METRICS = (
    "raw_resid_px",
    "d2_observation_only",
    "d2_predictive_raw",
    "d2_predictive_loso_sensitivity",
)
ABLATION_METRICS = (
    "d2_ablate_loso_shape_raw_scale",
    "d2_ablate_raw_shape_loso_scale",
    "d2_ablate_loso_scale_isotropic",
)
SOURCE_D2_METRICS = (
    "d2_source_geometry_only_matched",
    "d2_source_constant_scale",
    "d2_source_shuffled_scale",
)
SOURCE_SCALAR_METRICS = (
    "source_scale_loso_trace",
    "source_geometry_trace_LL",
    "source_depth_m",
    "source_radius2_norm",
    "source_scale_over_trace_LL",
)


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def finite(values):
    a = np.asarray(list(values), dtype=np.float64)
    return a[np.isfinite(a)]


def med(values):
    a = finite(values)
    return float(np.median(a)) if a.size else float("nan")


def mean(values):
    a = finite(values)
    return float(np.mean(a)) if a.size else float("nan")


def max_finite(values):
    a = finite(values)
    return float(np.max(a)) if a.size else float("nan")


def separability(auc):
    return max(float(auc), 1.0 - float(auc)) if np.isfinite(auc) else float("nan")


def summarize(payloads):
    ps = sorted(payloads, key=lambda p: int(p["frame"]))
    valid_methods = {
        "m1_motion_spatial_crossfit_shadow_v1",
        "m1_motion_spatial_crossfit_shadow_v2_scale_shape",
        "m1_motion_spatial_crossfit_shadow_v3_scale_source",
    }
    valid = [p for p in ps if p.get("method") in valid_methods]
    if not valid:
        raise ValueError("No valid M1 spatial cross-fit payloads")

    has_ablation = all(
        all(m in p.get("auc_dynamic_vs_static", {}) for m in ABLATION_METRICS)
        for p in valid
    )
    has_source = all(
        all(
            m in p.get("auc_dynamic_vs_static", {})
            for m in SOURCE_D2_METRICS + SOURCE_SCALAR_METRICS
        )
        for p in valid
    )

    metrics = BASE_METRICS
    if has_ablation:
        metrics += ABLATION_METRICS
    if has_source:
        metrics += SOURCE_D2_METRICS + SOURCE_SCALAR_METRICS

    auc = {
        m: med(p["auc_dynamic_vs_static"].get(m, np.nan) for p in valid)
        for m in metrics
    }

    def q50(group, metric):
        return med(p["pooled_groups"][group][metric]["q50"] for p in valid)

    out = {
        "n_frames": len(valid),
        "frame_min": int(valid[0]["frame"]),
        "frame_max": int(valid[-1]["frame"]),
        "has_scale_shape_ablation": bool(has_ablation),
        "has_scale_source_audit": bool(has_source),
        "median_frame_auc": auc,
        "frame_median_q50": {},
        "mean_invariant_raw_pred_gt_obs": mean(
            f["invariants"]["raw_pred_gt_obs_fraction"]
            for p in valid
            for f in p["folds"]
        ),
        "mean_invariant_loso_pred_gt_obs": mean(
            f["invariants"]["loso_pred_gt_obs_fraction"]
            for p in valid
            for f in p["folds"]
        ),
        "median_train_static_per_fold": med(
            f["n_train_static"] for p in valid for f in p["folds"]
        ),
        "median_eval_per_fold": med(
            f["n_eval"] for p in valid for f in p["folds"]
        ),
    }

    for metric in metrics:
        s = q50("baseline_static", metric)
        d = q50("baseline_dynamic", metric)
        out["frame_median_q50"][metric] = {
            "static": s,
            "dynamic": d,
            "dynamic_over_static": (
                d / s if np.isfinite(s) and s > 0 else float("nan")
            ),
        }

    for metric in (
        "pose_raw_to_observation_trace_ratio",
        "pose_loso_to_observation_trace_ratio",
    ):
        out["frame_median_q50"][metric] = {
            "static": q50("baseline_static", metric),
            "dynamic": q50("baseline_dynamic", metric),
        }

    if has_ablation:
        invariant_map = {
            "mean_invariant_loso_shape_raw_scale_pred_gt_obs":
                "loso_shape_raw_scale_pred_gt_obs_fraction",
            "mean_invariant_raw_shape_loso_scale_pred_gt_obs":
                "raw_shape_loso_scale_pred_gt_obs_fraction",
            "mean_invariant_loso_scale_isotropic_pred_gt_obs":
                "loso_scale_isotropic_pred_gt_obs_fraction",
            "max_trace_relerr_loso_shape_raw_scale":
                "loso_shape_raw_scale_trace_relerr_max",
            "max_trace_relerr_raw_shape_loso_scale":
                "raw_shape_loso_scale_trace_relerr_max",
            "max_trace_relerr_loso_scale_isotropic":
                "loso_scale_isotropic_trace_relerr_max",
        }
        for out_key, fold_key in invariant_map.items():
            values = [
                f["invariants"].get(fold_key, np.nan)
                for p in valid
                for f in p["folds"]
            ]
            out[out_key] = (
                max_finite(values) if out_key.startswith("max_") else mean(values)
            )

    if has_source:
        source_invariant_map = {
            "mean_invariant_source_geometry_pred_gt_obs":
                "source_geometry_pred_gt_obs_fraction",
            "mean_invariant_source_constant_pred_gt_obs":
                "source_constant_pred_gt_obs_fraction",
            "mean_invariant_source_shuffled_pred_gt_obs":
                "source_shuffled_pred_gt_obs_fraction",
        }
        for out_key, fold_key in source_invariant_map.items():
            out[out_key] = mean(
                f["invariants"].get(fold_key, np.nan)
                for p in valid
                for f in p["folds"]
            )
        out["max_source_geometry_median_match_abs_error"] = max_finite(
            abs(f["invariants"].get("source_geometry_median_match_ratio", np.nan) - 1.0)
            for p in valid
            for f in p["folds"]
        )
        out["max_source_constant_median_match_abs_error"] = max_finite(
            abs(f["invariants"].get("source_constant_median_match_ratio", np.nan) - 1.0)
            for p in valid
            for f in p["folds"]
        )
        out["max_source_shuffle_mean_relerr"] = max_finite(
            f["invariants"].get("source_shuffle_mean_relerr", np.nan)
            for p in valid
            for f in p["folds"]
        )
        out["source_scalar_separability"] = {
            m: separability(auc[m]) for m in SOURCE_SCALAR_METRICS
        }

    dystarts = [p.get("dystart") for p in valid if p.get("dystart") is not None]
    if dystarts:
        dystart = int(np.median(dystarts))
        out["dystart"] = dystart
        for radius in (10, 30):
            local = [
                p for p in valid if abs(int(p["frame"]) - dystart) <= radius
            ]
            if not local:
                continue
            local_auc = {
                m: med(p["auc_dynamic_vs_static"].get(m, np.nan) for p in local)
                for m in metrics
            }
            out[f"around_dystart_pm{radius}"] = {
                "n_frames": len(local),
                "median_auc": local_auc,
                "median_dynamic_over_static": {
                    m: (
                        med(
                            p["pooled_groups"]["baseline_dynamic"][m]["q50"]
                            for p in local
                        )
                        / med(
                            p["pooled_groups"]["baseline_static"][m]["q50"]
                            for p in local
                        )
                    )
                    for m in metrics
                },
                "source_scalar_separability": (
                    {
                        m: separability(local_auc[m])
                        for m in SOURCE_SCALAR_METRICS
                    }
                    if has_source
                    else {}
                ),
            }
    return out


def print_summary(label, s):
    print(f"\n=== {label} ===")
    print(f"frames={s['n_frames']} range={s['frame_min']}-{s['frame_max']}")
    print(
        "median train/eval pixels per fold: "
        f"{s['median_train_static_per_fold']:.0f} / "
        f"{s['median_eval_per_fold']:.0f}"
    )

    print("median frame AUC dynamic>static:")
    for k, v in s["median_frame_auc"].items():
        if k in SOURCE_SCALAR_METRICS:
            continue
        print(f"  {k:38s} {v:.4f}")

    if s.get("has_scale_source_audit"):
        print("source scalar AUC dynamic>static / direction-free separability:")
        for k in SOURCE_SCALAR_METRICS:
            auc = s["median_frame_auc"][k]
            sep = s["source_scalar_separability"][k]
            print(f"  {k:38s} {auc:.4f} / {sep:.4f}")

    print("frame-median q50 static / dynamic / ratio:")
    for k, v in s["frame_median_q50"].items():
        if "dynamic_over_static" in v:
            print(
                f"  {k:38s} {v['static']:.5g} / {v['dynamic']:.5g} / "
                f"{v['dynamic_over_static']:.3f}x"
            )
        else:
            print(f"  {k:38s} {v['static']:.5g} / {v['dynamic']:.5g}")

    print(
        "PSD invariant raw/LOSO pred_d2>obs_d2 mean fraction: "
        f"{s['mean_invariant_raw_pred_gt_obs']:.3e} / "
        f"{s['mean_invariant_loso_pred_gt_obs']:.3e}"
    )

    if s.get("has_scale_shape_ablation"):
        print(
            "PSD invariant shapeRaw/rawShape/isotropic mean fraction: "
            f"{s['mean_invariant_loso_shape_raw_scale_pred_gt_obs']:.3e} / "
            f"{s['mean_invariant_raw_shape_loso_scale_pred_gt_obs']:.3e} / "
            f"{s['mean_invariant_loso_scale_isotropic_pred_gt_obs']:.3e}"
        )
        print(
            "trace-match max relative error shapeRaw/rawShape/isotropic: "
            f"{s['max_trace_relerr_loso_shape_raw_scale']:.3e} / "
            f"{s['max_trace_relerr_raw_shape_loso_scale']:.3e} / "
            f"{s['max_trace_relerr_loso_scale_isotropic']:.3e}"
        )

    if s.get("has_scale_source_audit"):
        print(
            "PSD invariant source geometry/constant/shuffle mean fraction: "
            f"{s['mean_invariant_source_geometry_pred_gt_obs']:.3e} / "
            f"{s['mean_invariant_source_constant_pred_gt_obs']:.3e} / "
            f"{s['mean_invariant_source_shuffled_pred_gt_obs']:.3e}"
        )
        print(
            "source control invariant geometry-median/constant-median/shuffle-mean: "
            f"{s['max_source_geometry_median_match_abs_error']:.3e} / "
            f"{s['max_source_constant_median_match_abs_error']:.3e} / "
            f"{s['max_source_shuffle_mean_relerr']:.3e}"
        )

    if "dystart" in s:
        print(f"dystart={s['dystart']}")
        for radius in (10, 30):
            key = f"around_dystart_pm{radius}"
            if key not in s:
                continue
            x = s[key]
            print(f"  +/-{radius} frames={x['n_frames']}")
            for m in s["median_frame_auc"]:
                if m == "raw_resid_px" or m in SOURCE_SCALAR_METRICS:
                    continue
                print(
                    f"    {m:36s} AUC={x['median_auc'][m]:.4f} "
                    f"dyn/static={x['median_dynamic_over_static'][m]:.3f}x"
                )
            if s.get("has_scale_source_audit"):
                print("    source scalar AUC / separability:")
                for m in SOURCE_SCALAR_METRICS:
                    print(
                        f"      {m:34s} "
                        f"{x['median_auc'][m]:.4f} / "
                        f"{x['source_scalar_separability'][m]:.4f}"
                    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    report = {
        "method": "m1_motion_spatial_crossfit_audit_v3_scale_source",
        "note": (
            "No intervention threshold selected; LOSO calibration is "
            "sensitivity-only. V3 source controls are label-free constructions "
            "inside each held-out fold. Scalar AUCs use baseline motion partition "
            "only for diagnostic association, not segmentation ground truth."
        ),
        "runs": {},
    }

    for run in args.run_dirs:
        run = Path(run)
        files = sorted((run / "m1_motion_crossfit").glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No m1_motion_crossfit/*.pt under {run}")
        payloads = [load(p) for p in files]
        s = summarize(payloads)
        report["runs"][run.name] = s
        print_summary(run.name, s)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=True))
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
