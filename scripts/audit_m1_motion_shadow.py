#!/usr/bin/env python3
"""Aggregate M1 probabilistic motion-mask shadow diagnostics.

The audit intentionally does not choose an intervention threshold. The chi^2
q95/q99 values saved by the runtime are reference points only; M1 residuals are
not assumed to be exactly independent chi-square samples because the camera
motion is fitted from overlapping image evidence.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _nanmedian(values):
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _nanmean(values):
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _weighted_fraction(payloads, field_path, weight_path):
    num = 0.0
    den = 0.0
    for p in payloads:
        value = p
        for key in field_path:
            value = value[key]
        weight = p
        for key in weight_path:
            weight = weight[key]
        if np.isfinite(value) and weight > 0:
            num += float(value) * float(weight)
            den += float(weight)
    return num / den if den > 0 else float("nan")


def _subset(payloads, lo=None, hi=None):
    out = []
    for p in payloads:
        frame = int(p["frame"])
        if lo is not None and frame < lo:
            continue
        if hi is not None and frame > hi:
            continue
        out.append(p)
    return out


def summarize(payloads):
    payloads = sorted(payloads, key=lambda p: int(p["frame"]))
    frames = [int(p["frame"]) for p in payloads]
    valid = [
        p
        for p in payloads
        if int(p.get("n_valid", 0)) > 0 and p.get("covariance_mode") == "cluster"
    ]
    if not valid:
        return {
            "n_frames": len(payloads),
            "n_valid_frames": 0,
            "n_noncluster_frames_skipped": len(payloads),
        }

    def q50(group, metric):
        return _nanmedian(
            p["groups"][group][metric]["q50"] for p in valid
        )

    result = {
        "n_frames": len(payloads),
        "n_valid_frames": len(valid),
        "n_noncluster_frames_skipped": len(payloads) - len(valid),
        "frame_min": min(frames),
        "frame_max": max(frames),
        "total_valid_pixels": int(sum(int(p["n_valid"]) for p in valid)),
        "median_baseline_dynamic_fraction": _nanmedian(
            p["baseline_dynamic_fraction"] for p in valid
        ),
        "frame_median_raw_resid_px": {
            "baseline_static": q50("baseline_static", "raw_resid_px"),
            "baseline_dynamic": q50("baseline_dynamic", "raw_resid_px"),
        },
        "frame_median_d2_observation_only": {
            "baseline_static": q50("baseline_static", "d2_observation_only"),
            "baseline_dynamic": q50("baseline_dynamic", "d2_observation_only"),
        },
        "frame_median_d2_pose_predictive": {
            "baseline_static": q50("baseline_static", "d2_pose_predictive"),
            "baseline_dynamic": q50("baseline_dynamic", "d2_pose_predictive"),
        },
        "frame_median_pose_to_observation_trace_ratio": {
            "baseline_static": q50(
                "baseline_static", "pose_to_observation_trace_ratio"
            ),
            "baseline_dynamic": q50(
                "baseline_dynamic", "pose_to_observation_trace_ratio"
            ),
        },
        "invariant_predictive_gt_observation_mean_fraction": _nanmean(
            p["invariants"]["predictive_d2_gt_observation_d2_fraction"]
            for p in valid
        ),
    }

    for level in ("q95", "q99"):
        result[f"{level}_baseline_vs_predictive_disagree_fraction"] = _weighted_fraction(
            valid,
            ("disagreement", level, "baseline_vs_predictive_disagree_fraction"),
            ("n_valid",),
        )
        result[f"{level}_baseline_dynamic_to_predictive_static_fraction"] = _weighted_fraction(
            valid,
            ("disagreement", level, "baseline_dynamic_to_predictive_static_fraction"),
            ("n_baseline_dynamic",),
        )
        result[f"{level}_baseline_static_to_predictive_dynamic_fraction"] = _weighted_fraction(
            valid,
            ("disagreement", level, "baseline_static_to_predictive_dynamic_fraction"),
            ("n_baseline_static",),
        )
        result[f"{level}_observation_dynamic_fraction"] = _weighted_fraction(
            valid,
            ("disagreement", level, "observation_dynamic_fraction"),
            ("n_valid",),
        )
        result[f"{level}_predictive_dynamic_fraction"] = _weighted_fraction(
            valid,
            ("disagreement", level, "predictive_dynamic_fraction"),
            ("n_valid",),
        )

    dystarts = [
        p.get("dystart") for p in valid if p.get("dystart") is not None
    ]
    if dystarts:
        dystart = int(np.median(np.asarray(dystarts)))
        result["dystart"] = dystart
        for radius in (10, 30):
            local = _subset(valid, dystart - radius, dystart + radius)
            if local:
                result[f"around_dystart_pm{radius}"] = {
                    "n_frames": len(local),
                    "median_baseline_dynamic_fraction": _nanmedian(
                        p["baseline_dynamic_fraction"] for p in local
                    ),
                    "q95_disagree_fraction": _weighted_fraction(
                        local,
                        ("disagreement", "q95", "baseline_vs_predictive_disagree_fraction"),
                        ("n_valid",),
                    ),
                    "q95_dynamic_to_static_fraction": _weighted_fraction(
                        local,
                        (
                            "disagreement",
                            "q95",
                            "baseline_dynamic_to_predictive_static_fraction",
                        ),
                        ("n_baseline_dynamic",),
                    ),
                }

    ranked = sorted(
        valid,
        key=lambda p: (
            -1.0
            if not np.isfinite(
                p["disagreement"]["q95"][
                    "baseline_dynamic_to_predictive_static_fraction"
                ]
            )
            else p["disagreement"]["q95"][
                "baseline_dynamic_to_predictive_static_fraction"
            ]
        ),
        reverse=True,
    )[:10]
    result["top_q95_dynamic_to_static_frames"] = [
        {
            "frame": int(p["frame"]),
            "fraction": float(
                p["disagreement"]["q95"][
                    "baseline_dynamic_to_predictive_static_fraction"
                ]
            ),
            "baseline_dynamic_fraction": float(p["baseline_dynamic_fraction"]),
            "pose_to_obs_q50_dynamic": float(
                p["groups"]["baseline_dynamic"][
                    "pose_to_observation_trace_ratio"
                ]["q50"]
            ),
        }
        for p in ranked
        if np.isfinite(
            p["disagreement"]["q95"][
                "baseline_dynamic_to_predictive_static_fraction"
            ]
        )
    ]
    return result


def print_summary(label, summary):
    print(f"\n=== {label} ===")
    print(
        f"frames={summary.get('n_valid_frames', 0)}/{summary.get('n_frames', 0)} "
        f"valid_pixels={summary.get('total_valid_pixels', 0)} "
        f"noncluster_skipped={summary.get('n_noncluster_frames_skipped', 0)}"
    )
    if summary.get("n_valid_frames", 0) == 0:
        return
    print(
        "frame-median d2 obs static/dynamic: "
        f"{summary['frame_median_d2_observation_only']['baseline_static']:.4g} / "
        f"{summary['frame_median_d2_observation_only']['baseline_dynamic']:.4g}"
    )
    print(
        "frame-median d2 pred static/dynamic: "
        f"{summary['frame_median_d2_pose_predictive']['baseline_static']:.4g} / "
        f"{summary['frame_median_d2_pose_predictive']['baseline_dynamic']:.4g}"
    )
    print(
        "frame-median pose/obs trace static/dynamic: "
        f"{summary['frame_median_pose_to_observation_trace_ratio']['baseline_static']:.4g} / "
        f"{summary['frame_median_pose_to_observation_trace_ratio']['baseline_dynamic']:.4g}"
    )
    for level in ("q95", "q99"):
        print(
            f"{level}: disagree="
            f"{summary[level + '_baseline_vs_predictive_disagree_fraction']:.3%}, "
            f"baseline dynamic->pred static="
            f"{summary[level + '_baseline_dynamic_to_predictive_static_fraction']:.3%}, "
            f"baseline static->pred dynamic="
            f"{summary[level + '_baseline_static_to_predictive_dynamic_fraction']:.3%}"
        )
    print(
        "PSD invariant pred_d2>obs_d2 mean fraction: "
        f"{summary['invariant_predictive_gt_observation_mean_fraction']:.3e}"
    )
    if "dystart" in summary:
        print(f"dystart={summary['dystart']}")
        for radius in (10, 30):
            key = f"around_dystart_pm{radius}"
            if key in summary:
                local = summary[key]
                print(
                    f"  +/-{radius}: q95 disagree={local['q95_disagree_fraction']:.3%}, "
                    f"dynamic->static={local['q95_dynamic_to_static_fraction']:.3%}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="SLAM result directories containing m1_motion_shadow/*.pt",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report = {
        "method": "m1_probabilistic_motion_shadow_audit_v1",
        "note": (
            "chi-square q95/q99 are diagnostic reference thresholds only; "
            "no runtime intervention threshold is selected by this audit"
        ),
        "runs": {},
    }

    for run in args.run_dirs:
        run_dir = Path(run)
        files = sorted((run_dir / "m1_motion_shadow").glob("*.pt"))
        if not files:
            raise FileNotFoundError(
                f"No m1_motion_shadow/*.pt under {run_dir}"
            )
        payloads = [_load_payload(path) for path in files]
        bad = [
            p.get("method")
            for p in payloads
            if p.get("method") != "m1_probabilistic_motion_shadow_v1"
        ]
        if bad:
            raise ValueError(
                f"Unexpected payload methods under {run_dir}: {sorted(set(bad))}"
            )
        summary = summarize(payloads)
        label = run_dir.name
        report["runs"][label] = summary
        print_summary(label, summary)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(report, f, indent=2, allow_nan=True)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
