#!/usr/bin/env python3
"""Audit pose corrections injected between M2-A tracking steps.

The backend optimizes keyframe poses and frontend.sync_backend() copies the
updated R/T into frontend cameras.  The M2 shadow covariance stored on those
Camera objects is not updated by that sync.

For current frame k, the saved quantities satisfy
    T_motion_prior[k] = T_prev_used[k] @ T_rel_applied[k].
Therefore
    T_prev_used[k] = T_motion_prior[k] @ inv(T_rel_applied[k])
recovers the previous-frame mean actually used to seed frame k.  Comparing it
with the previous frame's saved T_final reveals any pose correction injected
between the two tracking steps (normally backend BA synchronization).

This is an offline diagnostic only; it does not change SLAM runtime.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.m2_uncertainty import SE3_log


def parse_specs(values):
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=RUN_DIR, got {value!r}")
        name, path = value.split("=", 1)
        out.append((name.strip(), Path(path).expanduser()))
    return out


def pose_from(d, key):
    x = d.get(key)
    if not isinstance(x, torch.Tensor):
        return None
    x = x.double()
    if tuple(x.shape) != (4, 4) or not bool(torch.isfinite(x).all()):
        return None
    return x


def rotation_angle(T):
    c = torch.clamp((torch.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(torch.acos(c))


def pose_delta_stats(Ta, Tb):
    """Magnitude of left correction L such that Tb = L Ta."""
    L = Tb @ torch.linalg.inv(Ta)
    return float(torch.linalg.norm(L[:3, 3])), rotation_angle(L), L


def right_error(T, G):
    return SE3_log(torch.linalg.inv(T) @ G)


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def audit_run(name, run_dir):
    m2_dir = run_dir / "m2a_pose_uncertainty"
    m1_dir = run_dir / "m1_pose_uncertainty"
    if not m2_dir.is_dir():
        raise FileNotFoundError(m2_dir)
    if not m1_dir.is_dir():
        raise FileNotFoundError(
            f"{m1_dir} is required because T_rel_applied is saved by M1"
        )

    m2 = {}
    for p in sorted(m2_dir.glob("*.pt")):
        d = torch.load(p, map_location="cpu")
        frame = int(d.get("frame", int(p.stem)))
        m2[frame] = d

    m1 = {}
    for p in sorted(m1_dir.glob("*.pt")):
        d = torch.load(p, map_location="cpu")
        frame = int(d.get("frame", int(p.stem)))
        m1[frame] = d

    frames = sorted(set(m2) & set(m1))
    frame_set = set(m2)
    rows = []

    for frame in frames:
        # The M1 relative motion is from the immediately preceding processed
        # camera. Infer that predecessor as the largest saved M2 frame < k.
        prev_candidates = [f for f in frame_set if f < frame]
        if not prev_candidates:
            continue
        prev_frame = max(prev_candidates)

        cur = m2[frame]
        prev = m2[prev_frame]
        rel = m1[frame]

        T_prior = pose_from(cur, "T_motion_prior")
        T_rel = pose_from(rel, "T_rel_applied")
        T_prev_saved = pose_from(prev, "T_final")
        G_prev = pose_from(prev, "T_gt")
        G_cur = pose_from(cur, "T_gt")
        if any(x is None for x in (
            T_prior, T_rel, T_prev_saved, G_prev, G_cur
        )):
            continue

        # Exact algebraic recovery under the saved frontend composition.
        T_prev_used = T_prior @ torch.linalg.inv(T_rel)
        corr_t, corr_r, _ = pose_delta_stats(T_prev_saved, T_prev_used)

        e_saved = right_error(T_prev_saved, G_prev)
        e_used = right_error(T_prev_used, G_prev)
        e_prior = right_error(T_prior, G_cur)

        # Counterfactual diagnostic: compose the same relative motion from the
        # previous tracking-time mean, i.e. without the inter-step correction.
        T_prior_no_interstep = T_prev_saved @ T_rel
        e_prior_no = right_error(T_prior_no_interstep, G_cur)

        P_prev = prev.get("P_abs_right")
        sigma_t = sigma_r = float("nan")
        if isinstance(P_prev, torch.Tensor):
            P_prev = 0.5 * (P_prev.double() + P_prev.double().T)
            if bool(torch.isfinite(P_prev).all()):
                d = torch.diagonal(P_prev).clamp_min(0.0)
                sigma_t = float(torch.sqrt(d[:3].mean()))
                sigma_r = float(torch.sqrt(d[3:].mean()))

        rows.append({
            "frame": frame,
            "prev_frame": prev_frame,
            "correction_t": corr_t,
            "correction_r": corr_r,
            "prev_error_saved_t": float(torch.linalg.norm(e_saved[:3])),
            "prev_error_saved_r": float(torch.linalg.norm(e_saved[3:])),
            "prev_error_used_t": float(torch.linalg.norm(e_used[:3])),
            "prev_error_used_r": float(torch.linalg.norm(e_used[3:])),
            "prior_error_actual_t": float(torch.linalg.norm(e_prior[:3])),
            "prior_error_actual_r": float(torch.linalg.norm(e_prior[3:])),
            "prior_error_no_interstep_t": float(
                torch.linalg.norm(e_prior_no[:3])
            ),
            "prior_error_no_interstep_r": float(
                torch.linalg.norm(e_prior_no[3:])
            ),
            "prev_sigma_t": sigma_t,
            "prev_sigma_r": sigma_r,
        })

    if not rows:
        raise RuntimeError(f"No matched transitions for {name}")

    def col(key):
        return [r[key] for r in rows if math.isfinite(r[key])]

    corr_t = col("correction_t")
    corr_r = col("correction_r")
    ratios_t = [
        r["correction_t"] / r["prev_sigma_t"]
        for r in rows
        if math.isfinite(r["prev_sigma_t"]) and r["prev_sigma_t"] > 0
    ]
    ratios_r = [
        r["correction_r"] / r["prev_sigma_r"]
        for r in rows
        if math.isfinite(r["prev_sigma_r"]) and r["prev_sigma_r"] > 0
    ]

    # Positive means the inter-step update increased raw GT error.
    prev_delta_t = [
        r["prev_error_used_t"] - r["prev_error_saved_t"] for r in rows
    ]
    prev_delta_r = [
        r["prev_error_used_r"] - r["prev_error_saved_r"] for r in rows
    ]
    prior_delta_t = [
        r["prior_error_actual_t"] - r["prior_error_no_interstep_t"]
        for r in rows
    ]
    prior_delta_r = [
        r["prior_error_actual_r"] - r["prior_error_no_interstep_r"]
        for r in rows
    ]

    return {
        "name": name,
        "run_dir": str(run_dir),
        "transitions": len(rows),
        "interstep_pose_correction_translation_m": stats(corr_t),
        "interstep_pose_correction_rotation_rad": stats(corr_r),
        "correction_over_prev_sigma_translation": stats(ratios_t),
        "correction_over_prev_sigma_rotation": stats(ratios_r),
        "change_in_prev_raw_gt_error_translation_m": stats(prev_delta_t),
        "change_in_prev_raw_gt_error_rotation_rad": stats(prev_delta_r),
        "change_in_current_prior_error_vs_no_interstep_translation_m": stats(
            prior_delta_t
        ),
        "change_in_current_prior_error_vs_no_interstep_rotation_rad": stats(
            prior_delta_r
        ),
        "fraction_correction_gt_1cm": float(
            np.mean(np.asarray(corr_t) > 0.01)
        ),
        "fraction_correction_gt_1deg": float(
            np.mean(np.asarray(corr_r) > np.deg2rad(1.0))
        ),
        "note": (
            "Inter-step pose corrections are inferred algebraically from "
            "saved tracking/prior states. In normal execution these are "
            "primarily backend keyframe pose updates synchronized into the "
            "frontend. The counterfactual no-interstep comparison is "
            "diagnostic, not a causal intervention."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="+", required=True, metavar="NAME=RUN_DIR")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/m2a_backend_sync_audit.json"),
    )
    args = p.parse_args()

    report = {}
    print("=" * 84)
    print("M2-A inter-step/backend pose-correction audit")
    print("=" * 84)
    for name, run_dir in parse_specs(args.run):
        r = audit_run(name, run_dir)
        report[name] = r
        print(f"\n{name}: {r['transitions']} transitions")
        a = r["interstep_pose_correction_translation_m"]
        b = r["interstep_pose_correction_rotation_rad"]
        print(
            "  inter-step correction median/p95: "
            f"{a['median']:.6g}/{a['p95']:.6g} m, "
            f"{b['median']:.6g}/{b['p95']:.6g} rad"
        )
        a = r["correction_over_prev_sigma_translation"]
        b = r["correction_over_prev_sigma_rotation"]
        print(
            "  correction / previous sigma median/p95: "
            f"t={a['median']:.4g}/{a['p95']:.4g}, "
            f"r={b['median']:.4g}/{b['p95']:.4g}"
        )
        a = r["change_in_prev_raw_gt_error_translation_m"]
        b = r["change_in_prev_raw_gt_error_rotation_rad"]
        print(
            "  change in previous raw GT error (used - saved) median: "
            f"t={a['median']:.6g} m, r={b['median']:.6g} rad"
        )
        a = r[
            "change_in_current_prior_error_vs_no_interstep_translation_m"
        ]
        b = r[
            "change_in_current_prior_error_vs_no_interstep_rotation_rad"
        ]
        print(
            "  current prior error change vs no-interstep median: "
            f"t={a['median']:.6g} m, r={b['median']:.6g} rad"
        )
        print(
            "  fraction corrections >1cm / >1deg: "
            f"{100*r['fraction_correction_gt_1cm']:.2f}% / "
            f"{100*r['fraction_correction_gt_1deg']:.2f}%"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
