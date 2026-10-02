#!/usr/bin/env python3
"""Temporal concentration-vs-divergence audit for M1 mapping v1.

Uses runtime mapping weights from the corrected failure audit, but loads the full
per-frame M2-A diagnostic trajectories directly from each run directory. This
avoids the severe sampling bias of restricting pose comparison to frames that
happen to occur in both runs' mapping windows.

Diagnostic only: mapping calls overlap and are temporally dependent, and the two
runs are independently gauge-aligned. Associations below describe temporal
ordering; they do not establish causality.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.m2_uncertainty import SE3_log


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
    xx, yy = zip(*pairs)
    rx, ry = rankdata(xx), rankdata(yy)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan"), len(pairs)
    return float(np.corrcoef(rx, ry)[0, 1]), len(pairs)


def quantiles(values):
    x = np.asarray([float(v) for v in values if finite(v)], dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "q05": float(np.quantile(x, .05)),
        "q25": float(np.quantile(x, .25)), "median": float(np.median(x)),
        "q75": float(np.quantile(x, .75)), "q95": float(np.quantile(x, .95)),
        "max": float(x.max()),
    }


def fit_se3_alignment(est_xyz, gt_xyz):
    xbar, ybar = est_xyz.mean(axis=0), gt_xyz.mean(axis=0)
    X, Y = est_xyz - xbar, gt_xyz - ybar
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = ybar - R @ xbar
    A = np.eye(4, dtype=np.float64)
    A[:3, :3], A[:3, 3] = R, t
    return A


def load_full_pose_trajectory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"M2-A diagnostic directory does not exist: {directory}")
    samples = []
    for path in sorted(directory.glob("*.pt")):
        d = torch.load(path, map_location="cpu")
        T, G = d.get("T_final"), d.get("T_gt")
        if not isinstance(T, torch.Tensor) or not isinstance(G, torch.Tensor):
            continue
        if tuple(T.shape) != (4, 4) or tuple(G.shape) != (4, 4):
            continue
        T, G = T.double(), G.double()
        if not bool(torch.isfinite(T).all() and torch.isfinite(G).all()):
            continue
        samples.append({
            "frame": int(d.get("frame", int(path.stem))),
            "T": T,
            "G": G,
        })
    if len(samples) < 10:
        raise RuntimeError(f"Only {len(samples)} usable full-pose diagnostics in {directory}")

    est_xyz = np.stack([torch.linalg.inv(s["T"])[:3, 3].numpy() for s in samples])
    gt_xyz = np.stack([torch.linalg.inv(s["G"])[:3, 3].numpy() for s in samples])
    A_np = fit_se3_alignment(est_xyz, gt_xyz)
    A = torch.from_numpy(A_np).double()
    Ainv = torch.linalg.inv(A)

    out = {}
    for s in samples:
        T_aligned = s["T"] @ Ainv
        C = torch.linalg.inv(T_aligned)[:3, 3]
        Cg = torch.linalg.inv(s["G"])[:3, 3]
        e = SE3_log(torch.linalg.inv(T_aligned) @ s["G"])
        out[s["frame"]] = {
            "pose_center_error_m": float(torch.linalg.norm(C - Cg)),
            "pose_log_translation_error_m": float(torch.linalg.norm(e[:3])),
            "pose_rotation_error_rad": float(torch.linalg.norm(e[3:])),
        }
    return out, A_np.tolist(), len(samples)


def event_records(run_json):
    out = []
    for e in run_json["events"]:
        frames = e.get("frames", [])
        weights = e.get("actual_weights", [])
        if not frames or not weights:
            continue
        n = len(weights)
        ess = float(e["ess_actual"])
        out.append({
            "map_call": int(e["map_call"]),
            # Backend prints the current/newest keyframe first in the current window.
            "current_frame": int(frames[0]),
            "window_size": n,
            "ess": ess,
            "concentration": 1.0 - ess / n,
            "max_weight": float(max(weights)),
            "min_weight": float(min(weights)),
            "weight_std": float(np.std(np.asarray(weights, dtype=np.float64))),
        })
    return out


def frame_gap_table(shadow, weighted):
    rows = []
    for f in sorted(set(shadow) & set(weighted)):
        s, w = shadow[f], weighted[f]
        rows.append({
            "frame": f,
            "shadow_center_error_m": s["pose_center_error_m"],
            "weighted_center_error_m": w["pose_center_error_m"],
            "delta_center_error_m": w["pose_center_error_m"] - s["pose_center_error_m"],
            "shadow_rotation_error_rad": s["pose_rotation_error_rad"],
            "weighted_rotation_error_rad": w["pose_rotation_error_rad"],
            "delta_rotation_error_rad": w["pose_rotation_error_rad"] - s["pose_rotation_error_rad"],
        })
    return rows


def lag_audit(events, gap_rows, horizons, key):
    by_frame = {r["frame"]: r for r in gap_rows}
    result = {}
    for h in horizons:
        conc, ess, maxw, changes, levels = [], [], [], [], []
        for e in events:
            k = e["current_frame"]
            if k not in by_frame or k + h not in by_frame:
                continue
            now, future = by_frame[k][key], by_frame[k + h][key]
            if not finite(now) or not finite(future):
                continue
            conc.append(e["concentration"])
            ess.append(e["ess"])
            maxw.append(e["max_weight"])
            changes.append(float(future - now))
            levels.append(float(future))
        rc, n = spearman(conc, changes)
        re, _ = spearman(ess, changes)
        rw, _ = spearman(maxw, changes)
        rl, _ = spearman(conc, levels)
        result[str(h)] = {
            "n_mapping_calls": n,
            "spearman_concentration_vs_future_gap_change": rc,
            "spearman_ess_vs_future_gap_change": re,
            "spearman_max_weight_vs_future_gap_change": rw,
            "spearman_concentration_vs_future_gap_level": rl,
            "future_gap_change": quantiles(changes),
        }
    return result


def rolling_dose(events, gap_rows, histories, key):
    result = {}
    for h in histories:
        doses, gaps = [], []
        for row in gap_rows:
            f = row["frame"]
            recent = [e for e in events if f - h < e["current_frame"] <= f]
            if not recent or not finite(row[key]):
                continue
            doses.append(float(np.mean([e["concentration"] for e in recent])))
            gaps.append(float(row[key]))
        rho, n = spearman(doses, gaps)
        result[str(h)] = {
            "n_frames": n,
            "spearman_recent_concentration_vs_current_gap": rho,
            "recent_concentration": quantiles(doses),
        }
    return result


def strongest_gap_windows(rows, width, key, top_k=5):
    vals = {r["frame"]: float(r[key]) for r in rows if finite(r[key])}
    candidates = []
    for start in sorted(vals):
        x = [vals[f] for f in range(start, start + width) if f in vals]
        if len(x) >= max(3, int(.8 * width)):
            candidates.append((float(np.mean(x)), start, start + width - 1, len(x)))
    selected = []
    for mean_gap, start, end, n in sorted(candidates, reverse=True):
        if any(not (end < s[1] or start > s[2]) for s in selected):
            continue
        selected.append((mean_gap, start, end, n))
        if len(selected) == top_k:
            break
    return [{"start_frame": s, "end_frame": e, "n_frames": n, "mean_gap": g}
            for g, s, e, n in selected]


def annotate(rows, events, histories):
    for row in rows:
        f = row["frame"]
        for h in histories:
            recent = [e for e in events if f - h < e["current_frame"] <= f]
            if recent:
                row[f"recent_{h}f_mean_concentration"] = float(np.mean([e["concentration"] for e in recent]))
                row[f"recent_{h}f_min_ess"] = float(min(e["ess"] for e in recent))
                row[f"recent_{h}f_max_weight"] = float(max(e["max_weight"] for e in recent))
                row[f"recent_{h}f_mapping_calls"] = len(recent)
    return rows


def write_csv(path, rows):
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-json", type=Path, required=True)
    # Kept only so the previous command line remains accepted; full trajectories
    # are now loaded directly from the m2a_dir recorded in audit-json.
    p.add_argument("--audit-csv", type=Path, default=None)
    p.add_argument("--shadow-name", default="shadow")
    p.add_argument("--weighted-name", default="weighted")
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 40])
    p.add_argument("--history-frames", type=int, nargs="+", default=[10, 20, 40])
    p.add_argument("--gap-window", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("results/box1_temporal_audit_full.json"))
    args = p.parse_args()

    with args.audit_json.open() as f:
        audit = json.load(f)
    for name in (args.shadow_name, args.weighted_name):
        if name not in audit.get("runs", {}):
            raise KeyError(f"Run {name!r} missing from {args.audit_json}")

    sr = audit["runs"][args.shadow_name]
    wr = audit["runs"][args.weighted_name]
    shadow_pose, A_s, n_s = load_full_pose_trajectory(sr["m2a_dir"])
    weighted_pose, A_w, n_w = load_full_pose_trajectory(wr["m2a_dir"])
    gaps = frame_gap_table(shadow_pose, weighted_pose)
    if len(gaps) < 100:
        raise RuntimeError(
            f"Only {len(gaps)} common full-pose frames. Expected dense per-frame M2-A diagnostics; "
            "do not interpret a sparse temporal audit."
        )

    shadow_events, weighted_events = event_records(sr), event_records(wr)
    if any(abs(e["ess"] - e["window_size"]) > 1e-6 for e in shadow_events):
        raise RuntimeError("Shadow actual weights are not neutral")
    if not weighted_events:
        raise RuntimeError("No weighted mapping events")

    gaps = annotate(gaps, weighted_events, args.history_frames)
    center_key, rot_key = "delta_center_error_m", "delta_rotation_error_rad"
    report = {
        "method": "m1_mapping_temporal_concentration_divergence_audit_full_pose",
        "source_json": str(args.audit_json),
        "shadow_m2a_dir": sr["m2a_dir"], "weighted_m2a_dir": wr["m2a_dir"],
        "shadow_pose_frames": n_s, "weighted_pose_frames": n_w,
        "common_pose_frames": len(gaps), "weighted_mapping_calls": len(weighted_events),
        "alignment_shadow": A_s, "alignment_weighted": A_w,
        "guardrail": (
            "Descriptive temporal audit. Each trajectory is independently no-scale SE(3) gauge-aligned "
            "to GT over its full saved trajectory. Mapping calls overlap and are dependent; association "
            "and temporal precedence do not establish causality."
        ),
        "weighted_actual_concentration": {
            "ess": quantiles([e["ess"] for e in weighted_events]),
            "concentration": quantiles([e["concentration"] for e in weighted_events]),
            "max_weight": quantiles([e["max_weight"] for e in weighted_events]),
        },
        "trajectory_gap": {
            "center_error_gap_m_weighted_minus_shadow": quantiles([r[center_key] for r in gaps]),
            "rotation_error_gap_rad_weighted_minus_shadow": quantiles([r[rot_key] for r in gaps]),
        },
        "lag_center": lag_audit(weighted_events, gaps, args.horizons, center_key),
        "lag_rotation": lag_audit(weighted_events, gaps, args.horizons, rot_key),
        "rolling_center": rolling_dose(weighted_events, gaps, args.history_frames, center_key),
        "rolling_rotation": rolling_dose(weighted_events, gaps, args.history_frames, rot_key),
        "largest_center_gap_windows": strongest_gap_windows(gaps, args.gap_window, center_key),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeline = args.output.with_suffix(".csv")
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    write_csv(timeline, gaps)

    print("=" * 94)
    print("M1 mapping temporal concentration-divergence audit — FULL per-frame pose")
    print("=" * 94)
    print(f"pose frames shadow/weighted/common={n_s}/{n_w}/{len(gaps)}  weighted mapping calls={len(weighted_events)}")
    s = report["weighted_actual_concentration"]["ess"]
    print(f"weighted ACTUAL ESS q05/median/q95: {s['q05']:.3f} / {s['median']:.3f} / {s['q95']:.3f}")
    s = report["trajectory_gap"]["center_error_gap_m_weighted_minus_shadow"]
    print(f"center-error gap W-S [m] q05/median/q95: {s['q05']:.5f} / {s['median']:.5f} / {s['q95']:.5f}")

    print("\nLag audit: concentration at mapping frame k vs FUTURE CENTER-gap change")
    print(" horizon    n    rho(conc,change)   rho(ESS,change)   rho(max_w,change)   rho(conc,future_level)")
    for h in args.horizons:
        x = report["lag_center"][str(h)]
        print(f" {h:>7d} {x['n_mapping_calls']:>4d} {x['spearman_concentration_vs_future_gap_change']:>18.3f} "
              f"{x['spearman_ess_vs_future_gap_change']:>17.3f} {x['spearman_max_weight_vs_future_gap_change']:>20.3f} "
              f"{x['spearman_concentration_vs_future_gap_level']:>24.3f}")

    print("\nRolling dose: recent concentration vs current CENTER-error gap")
    print(" history    n       rho")
    for h in args.history_frames:
        x = report["rolling_center"][str(h)]
        print(f" {h:>7d} {x['n_frames']:>4d} {x['spearman_recent_concentration_vs_current_gap']:>9.3f}")

    print(f"\nLargest non-overlapping {args.gap_window}-frame mean center-gap windows:")
    for x in report["largest_center_gap_windows"]:
        print(f" frames {x['start_frame']:>4d}-{x['end_frame']:<4d} mean gap={x['mean_gap']:.5f} m n={x['n_frames']}")

    print(f"\nSaved JSON: {args.output}")
    print(f"Saved timeline CSV: {timeline}")


if __name__ == "__main__":
    main()
