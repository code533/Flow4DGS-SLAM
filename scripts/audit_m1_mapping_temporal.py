#!/usr/bin/env python3
"""Temporal audit for M1 uncertainty-aware mapping.

Consumes the corrected output of audit_m1_mapping_failure.py and asks a narrow
question: does strong runtime mapping-weight concentration temporally precede
larger weighted-vs-shadow pose-error gaps?

This script is diagnostic, not an inferential significance test. Mapping calls
and repeated keyframe appearances are temporally dependent, and shadow/weighted
runs are independently gauge-aligned. Reported correlations therefore describe
association and temporal ordering only; they do not establish causality.

The audit deliberately does NOT change SLAM, M1 calibration, clipping, or the
weight formula.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def rankdata(values):
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def spearman(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if finite(a) and finite(b)]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    xx, yy = map(np.asarray, zip(*pairs))
    rx, ry = rankdata(xx), rankdata(yy)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan"), len(pairs)
    return float(np.corrcoef(rx, ry)[0, 1]), len(pairs)


def quantiles(values):
    x = np.asarray([float(v) for v in values if finite(v)], dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def load_unique_pose_rows(path, run_name):
    """Return one pose/observation row per frame, checking duplicate consistency."""
    by_frame = {}
    fields = [
        "pose_center_error_m",
        "pose_log_translation_error_m",
        "pose_rotation_error_rad",
        "rgb_abs_median",
        "depth_abs_median_m",
    ]
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run") != run_name:
                continue
            frame = int(row["frame"])
            vals = {}
            for key in fields:
                if finite(row.get(key)):
                    vals[key] = float(row[key])
            if not vals:
                continue
            if frame in by_frame:
                # A frame can appear in many mapping windows. Its saved pose diagnostic
                # must be identical each time; fail rather than silently average conflicts.
                for key, value in vals.items():
                    if key in by_frame[frame] and abs(by_frame[frame][key] - value) > 1e-10:
                        raise RuntimeError(
                            f"Inconsistent {key} for {run_name} frame {frame}: "
                            f"{by_frame[frame][key]} vs {value}"
                        )
                by_frame[frame].update(vals)
            else:
                by_frame[frame] = vals
    if not by_frame:
        raise RuntimeError(f"No pose diagnostics found for run={run_name} in {path}")
    return by_frame


def event_records(run_json, use_actual):
    out = []
    for e in run_json["events"]:
        frames = e.get("frames", [])
        if not frames:
            continue
        current_frame = int(frames[0])
        ess_key = "ess_actual" if use_actual else "ess_candidate"
        ess = float(e[ess_key])
        weights = e["actual_weights"] if use_actual else e["candidate_weights"]
        n = len(weights)
        concentration = 1.0 - ess / max(1, n)
        out.append({
            "map_call": int(e["map_call"]),
            "current_frame": current_frame,
            "window_size": n,
            "ess": ess,
            "concentration": concentration,
            "max_weight": float(max(weights)),
            "min_weight": float(min(weights)),
            "weight_std": float(np.std(np.asarray(weights, dtype=np.float64))),
        })
    return out


def frame_gap_table(shadow_pose, weighted_pose):
    common = sorted(set(shadow_pose) & set(weighted_pose))
    rows = []
    for f in common:
        s, w = shadow_pose[f], weighted_pose[f]
        row = {"frame": f}
        for key in [
            "pose_center_error_m",
            "pose_log_translation_error_m",
            "pose_rotation_error_rad",
            "rgb_abs_median",
            "depth_abs_median_m",
        ]:
            if key in s and key in w:
                row[f"shadow_{key}"] = s[key]
                row[f"weighted_{key}"] = w[key]
                row[f"delta_{key}"] = w[key] - s[key]
        rows.append(row)
    return rows


def nearest_exact(gap_by_frame, frame, key):
    row = gap_by_frame.get(frame)
    if row is None or key not in row or not finite(row[key]):
        return None
    return float(row[key])


def lag_audit(events, gap_rows, horizons, delta_key):
    """Correlate treatment intensity at k with future change in error gap.

    future_change_h = delta_error(k+h) - delta_error(k). Using a change rather
    than only the future level reduces the trivial correlation caused by an
    already-diverged trajectory.
    """
    gap = {r["frame"]: r for r in gap_rows}
    result = {}
    for h in horizons:
        conc, ess_vals, maxw, future_change, future_level = [], [], [], [], []
        for e in events:
            k = e["current_frame"]
            now = nearest_exact(gap, k, delta_key)
            fut = nearest_exact(gap, k + h, delta_key)
            if now is None or fut is None:
                continue
            conc.append(e["concentration"])
            ess_vals.append(e["ess"])
            maxw.append(e["max_weight"])
            future_change.append(fut - now)
            future_level.append(fut)
        rho_c_change, n = spearman(conc, future_change)
        rho_ess_change, _ = spearman(ess_vals, future_change)
        rho_maxw_change, _ = spearman(maxw, future_change)
        rho_c_level, _ = spearman(conc, future_level)
        result[str(h)] = {
            "n_mapping_calls": n,
            "spearman_concentration_vs_future_gap_change": rho_c_change,
            "spearman_ess_vs_future_gap_change": rho_ess_change,
            "spearman_max_weight_vs_future_gap_change": rho_maxw_change,
            "spearman_concentration_vs_future_gap_level": rho_c_level,
        }
    return result


def rolling_dose_audit(events, gap_rows, history_frames, delta_key):
    """Associate recent cumulative mapping concentration with current error gap."""
    result = {}
    for h in history_frames:
        doses, gaps, frames = [], [], []
        for row in gap_rows:
            f = row["frame"]
            if delta_key not in row or not finite(row[delta_key]):
                continue
            recent = [e for e in events if f - h < e["current_frame"] <= f]
            if not recent:
                continue
            # Mean per optimization call intentionally preserves repeated mapping calls.
            dose = float(np.mean([e["concentration"] for e in recent]))
            doses.append(dose)
            gaps.append(float(row[delta_key]))
            frames.append(f)
        rho, n = spearman(doses, gaps)
        result[str(h)] = {
            "n_frames": n,
            "spearman_recent_concentration_vs_current_gap": rho,
            "recent_concentration": quantiles(doses),
        }
    return result


def strongest_gap_windows(gap_rows, width, delta_key, top_k=5):
    """Descriptive localization of contiguous frame windows with largest mean gap."""
    usable = {r["frame"]: float(r[delta_key]) for r in gap_rows if delta_key in r and finite(r[delta_key])}
    if not usable:
        return []
    frames = sorted(usable)
    candidates = []
    for start in frames:
        vals = [usable[f] for f in range(start, start + width) if f in usable]
        if len(vals) < max(3, int(0.8 * width)):
            continue
        candidates.append((float(np.mean(vals)), start, start + width - 1, len(vals)))
    # Greedy non-overlap keeps the report readable and avoids five near-identical windows.
    selected = []
    for mean_gap, start, end, n in sorted(candidates, reverse=True):
        if any(not (end < s[1] or start > s[2]) for s in selected):
            continue
        selected.append((mean_gap, start, end, n))
        if len(selected) == top_k:
            break
    return [
        {"start_frame": s, "end_frame": e, "n_frames": n, "mean_gap": g}
        for g, s, e, n in selected
    ]


def annotate_gap_rows(gap_rows, weighted_events, history_frames):
    for row in gap_rows:
        f = row["frame"]
        for h in history_frames:
            recent = [e for e in weighted_events if f - h < e["current_frame"] <= f]
            if recent:
                row[f"recent_{h}f_mean_concentration"] = float(np.mean([e["concentration"] for e in recent]))
                row[f"recent_{h}f_min_ess"] = float(min(e["ess"] for e in recent))
                row[f"recent_{h}f_max_weight"] = float(max(e["max_weight"] for e in recent))
                row[f"recent_{h}f_mapping_calls"] = len(recent)
    return gap_rows


def write_csv(path, rows):
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); keys.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--audit-csv", type=Path, required=True)
    p.add_argument("--shadow-name", default="shadow")
    p.add_argument("--weighted-name", default="weighted")
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 40])
    p.add_argument("--history-frames", type=int, nargs="+", default=[10, 20, 40])
    p.add_argument("--gap-window", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("results/box1_temporal_audit.json"))
    args = p.parse_args()

    with args.audit_json.open() as f:
        audit = json.load(f)
    for name in (args.shadow_name, args.weighted_name):
        if name not in audit.get("runs", {}):
            raise KeyError(f"Run {name!r} not found in {args.audit_json}")

    shadow_pose = load_unique_pose_rows(args.audit_csv, args.shadow_name)
    weighted_pose = load_unique_pose_rows(args.audit_csv, args.weighted_name)
    gaps = frame_gap_table(shadow_pose, weighted_pose)
    if len(gaps) < 10:
        raise RuntimeError(f"Only {len(gaps)} common pose frames; temporal audit is not meaningful")

    shadow_events = event_records(audit["runs"][args.shadow_name], use_actual=True)
    weighted_events = event_records(audit["runs"][args.weighted_name], use_actual=True)
    if not weighted_events:
        raise RuntimeError("No weighted mapping events")

    # Shadow diagnostic must truly be behavior-neutral at the loss weighting site.
    shadow_actual = [e["ess"] for e in shadow_events]
    shadow_sizes = [e["window_size"] for e in shadow_events]
    if any(abs(a - b) > 1e-6 for a, b in zip(shadow_actual, shadow_sizes)):
        raise RuntimeError("Shadow actual ESS is not equal to window size; shadow is not neutral")

    delta_center = "delta_pose_center_error_m"
    delta_rot = "delta_pose_rotation_error_rad"
    gaps = annotate_gap_rows(gaps, weighted_events, args.history_frames)

    report = {
        "method": "m1_mapping_temporal_concentration_divergence_audit",
        "source_json": str(args.audit_json),
        "source_csv": str(args.audit_csv),
        "shadow_name": args.shadow_name,
        "weighted_name": args.weighted_name,
        "common_pose_frames": len(gaps),
        "weighted_mapping_calls": len(weighted_events),
        "guardrail": (
            "Descriptive temporal audit only. Runs are independently gauge-aligned; "
            "mapping calls overlap and are not independent. Positive lag association "
            "is consistent with, but does not prove, a causal concentration->divergence mechanism."
        ),
        "weighted_actual_concentration": {
            "ess": quantiles([e["ess"] for e in weighted_events]),
            "concentration_1_minus_ess_over_n": quantiles([e["concentration"] for e in weighted_events]),
            "max_weight": quantiles([e["max_weight"] for e in weighted_events]),
        },
        "trajectory_gap": {
            "center_error_gap_m_weighted_minus_shadow": quantiles([
                r[delta_center] for r in gaps if delta_center in r
            ]),
            "rotation_error_gap_rad_weighted_minus_shadow": quantiles([
                r[delta_rot] for r in gaps if delta_rot in r
            ]),
        },
        "lag_audit_center_error": lag_audit(weighted_events, gaps, args.horizons, delta_center),
        "lag_audit_rotation_error": lag_audit(weighted_events, gaps, args.horizons, delta_rot),
        "rolling_dose_center_error": rolling_dose_audit(weighted_events, gaps, args.history_frames, delta_center),
        "rolling_dose_rotation_error": rolling_dose_audit(weighted_events, gaps, args.history_frames, delta_rot),
        "largest_center_gap_windows": strongest_gap_windows(gaps, args.gap_window, delta_center),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeline_csv = args.output.with_suffix(".csv")
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    write_csv(timeline_csv, gaps)

    print("=" * 92)
    print("M1 mapping temporal concentration-divergence audit")
    print("=" * 92)
    print(f"common pose frames={len(gaps)}  weighted mapping calls={len(weighted_events)}")
    s = report["weighted_actual_concentration"]["ess"]
    print(f"weighted ACTUAL ESS q05/median/q95: {s['q05']:.3f} / {s['median']:.3f} / {s['q95']:.3f}")
    s = report["trajectory_gap"]["center_error_gap_m_weighted_minus_shadow"]
    print(f"center-error gap W-S [m] q05/median/q95: {s['q05']:.5f} / {s['median']:.5f} / {s['q95']:.5f}")

    print("\nLag audit: concentration at mapping frame k vs future CENTER-error gap change")
    print("  horizon   n    rho(conc, future_change)   rho(ESS, future_change)   rho(max_w, future_change)")
    for h in args.horizons:
        x = report["lag_audit_center_error"][str(h)]
        print(f"  {h:>7d} {x['n_mapping_calls']:>4d} {x['spearman_concentration_vs_future_gap_change']:>26.3f} "
              f"{x['spearman_ess_vs_future_gap_change']:>25.3f} {x['spearman_max_weight_vs_future_gap_change']:>27.3f}")

    print("\nRolling dose: recent concentration vs current CENTER-error gap")
    print("  history   n    rho")
    for h in args.history_frames:
        x = report["rolling_dose_center_error"][str(h)]
        print(f"  {h:>7d} {x['n_frames']:>4d} {x['spearman_recent_concentration_vs_current_gap']:>7.3f}")

    print(f"\nLargest {args.gap_window}-frame mean center-error-gap windows:")
    for x in report["largest_center_gap_windows"]:
        print(f"  frames {x['start_frame']:>4d}-{x['end_frame']:<4d}  mean gap={x['mean_gap']:.5f} m  n={x['n_frames']}")

    print(f"\nSaved JSON: {args.output}")
    print(f"Saved timeline CSV: {timeline_csv}")


if __name__ == "__main__":
    main()
