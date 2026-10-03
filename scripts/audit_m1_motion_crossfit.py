#!/usr/bin/env python3
"""Aggregate spatial cross-fitted M1 motion-shadow diagnostics.

No intervention threshold is selected. LOSO-calibrated covariance is reported
only as a sensitivity analysis because the calibration was not trained for the
cross-fitted estimator.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


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


def summarize(payloads):
    ps = sorted(payloads, key=lambda p: int(p["frame"]))
    valid = [p for p in ps if p.get("method") == "m1_motion_spatial_crossfit_shadow_v1"]
    if not valid:
        raise ValueError("No valid m1_motion_spatial_crossfit_shadow_v1 payloads")

    metrics = (
        "raw_resid_px",
        "d2_observation_only",
        "d2_predictive_raw",
        "d2_predictive_loso_sensitivity",
    )
    auc = {m: med(p["auc_dynamic_vs_static"].get(m, np.nan) for p in valid) for m in metrics}

    def q50(group, metric):
        return med(p["pooled_groups"][group][metric]["q50"] for p in valid)

    out = {
        "n_frames": len(valid),
        "frame_min": int(valid[0]["frame"]),
        "frame_max": int(valid[-1]["frame"]),
        "median_frame_auc": auc,
        "frame_median_q50": {},
        "mean_invariant_raw_pred_gt_obs": mean(
            f["invariants"]["raw_pred_gt_obs_fraction"]
            for p in valid for f in p["folds"]
        ),
        "mean_invariant_loso_pred_gt_obs": mean(
            f["invariants"]["loso_pred_gt_obs_fraction"]
            for p in valid for f in p["folds"]
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
            "dynamic_over_static": d / s if np.isfinite(s) and s > 0 else float("nan"),
        }

    for metric in (
        "pose_raw_to_observation_trace_ratio",
        "pose_loso_to_observation_trace_ratio",
    ):
        out["frame_median_q50"][metric] = {
            "static": q50("baseline_static", metric),
            "dynamic": q50("baseline_dynamic", metric),
        }

    dystarts = [p.get("dystart") for p in valid if p.get("dystart") is not None]
    if dystarts:
        dystart = int(np.median(dystarts))
        out["dystart"] = dystart
        for radius in (10, 30):
            local = [p for p in valid if abs(int(p["frame"]) - dystart) <= radius]
            if not local:
                continue
            out[f"around_dystart_pm{radius}"] = {
                "n_frames": len(local),
                "median_auc": {
                    m: med(p["auc_dynamic_vs_static"].get(m, np.nan) for p in local)
                    for m in metrics
                },
                "median_dynamic_over_static": {
                    m: (
                        med(p["pooled_groups"]["baseline_dynamic"][m]["q50"] for p in local)
                        / med(p["pooled_groups"]["baseline_static"][m]["q50"] for p in local)
                    )
                    for m in metrics
                },
            }
    return out


def print_summary(label, s):
    print(f"\n=== {label} ===")
    print(f"frames={s['n_frames']} range={s['frame_min']}-{s['frame_max']}")
    print(
        "median train/eval pixels per fold: "
        f"{s['median_train_static_per_fold']:.0f} / {s['median_eval_per_fold']:.0f}"
    )
    print("median frame AUC dynamic>static:")
    for k, v in s["median_frame_auc"].items():
        print(f"  {k:35s} {v:.4f}")
    print("frame-median q50 static / dynamic / ratio:")
    for k, v in s["frame_median_q50"].items():
        if "dynamic_over_static" in v:
            print(f"  {k:35s} {v['static']:.5g} / {v['dynamic']:.5g} / {v['dynamic_over_static']:.3f}x")
        else:
            print(f"  {k:35s} {v['static']:.5g} / {v['dynamic']:.5g}")
    print(
        "PSD invariant raw/LOSO pred_d2>obs_d2 mean fraction: "
        f"{s['mean_invariant_raw_pred_gt_obs']:.3e} / "
        f"{s['mean_invariant_loso_pred_gt_obs']:.3e}"
    )
    if "dystart" in s:
        print(f"dystart={s['dystart']}")
        for radius in (10, 30):
            key = f"around_dystart_pm{radius}"
            if key not in s:
                continue
            x = s[key]
            print(f"  +/-{radius} frames={x['n_frames']}")
            for m in ("d2_observation_only", "d2_predictive_raw", "d2_predictive_loso_sensitivity"):
                print(
                    f"    {m:33s} AUC={x['median_auc'][m]:.4f} "
                    f"dyn/static={x['median_dynamic_over_static'][m]:.3f}x"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    report = {
        "method": "m1_motion_spatial_crossfit_audit_v1",
        "note": "No intervention threshold selected; LOSO calibration is sensitivity-only for this estimator.",
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
