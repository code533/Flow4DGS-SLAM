#!/usr/bin/env python3
"""Audit runtime M1 mapping weights against pose/observation consistency.

This is diagnostic-only. It parses the existing Backend log lines

    M1 RGB-D weights frame:weight, ...

and reconstructs the *candidate* uncertainty weights from saved M1 covariance
files. It can therefore audit both:

  * weighted runs, where the logged weights are applied; and
  * shadow-diagnostic runs, where runtime clipping is fixed to [1, 1] so the
    applied loss weight is exactly one while the same M1 signal path executes.

The script also joins saved M2-A/M2-A3 diagnostics when available. Pose error is
reported only after one no-scale SE(3) trajectory gauge alignment per run.
Observation-consistency values are diagnostics, not ground-truth mapping loss.
Repeated keyframe appearances across mapping windows are temporally dependent,
so correlations are descriptive rather than independent-sample tests.
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.m1_mapping_uncertainty import M1MappingSignal, normalized_window_weights
from utils.m2_uncertainty import SE3_log


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
WEIGHT_RE = re.compile(r"(-?\d+)\s*:\s*([0-9.eE+\-]+)")

OBS_KEYS = [
    "rgb_abs_median",
    "rgb_abs_p95",
    "depth_abs_median_m",
    "depth_abs_p95_m",
    "depth_rel_median",
    "rgb_inlier_ratio_0p05",
    "rgb_inlier_ratio_0p10",
    "depth_inlier_ratio_0p05m",
    "depth_inlier_ratio_0p10m",
    "rgb_opacity_coverage",
    "depth_opacity_coverage",
    "tracking_loss_relative_improvement",
]


def parse_run_spec(value):
    if "=" not in value:
        raise ValueError(
            "Expected NAME=LOG,M1_DIR,M2A_DIR for --run, got " + repr(value)
        )
    name, rest = value.split("=", 1)
    parts = [Path(x).expanduser() for x in rest.split(",")]
    if len(parts) != 3:
        raise ValueError(
            "Expected NAME=LOG,M1_DIR,M2A_DIR for --run, got " + repr(value)
        )
    return name.strip(), parts[0], parts[1], parts[2]


def finite(x):
    return math.isfinite(float(x))


def stats(values):
    x = np.asarray([float(v) for v in values if finite(v)], dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "q05": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)),
        "q95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


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
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if finite(a) and finite(b)
    ]
    if len(pairs) < 3:
        return float("nan")
    xx, yy = zip(*pairs)
    rx, ry = rankdata(xx), rankdata(yy)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def ess(weights):
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0 or not np.isfinite(w).all():
        return float("nan")
    denom = float(np.sum(w * w))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w) ** 2 / denom)


def parse_weight_log(path):
    events = []
    with path.open("r", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            line = ANSI_RE.sub("", raw)
            if "M1 RGB-D weights" not in line:
                continue
            tail = line.split("M1 RGB-D weights", 1)[1]
            pairs = WEIGHT_RE.findall(tail)
            if not pairs:
                continue
            events.append(
                {
                    "map_call": len(events) + 1,
                    "log_line": line_no,
                    "frames": [int(k) for k, _ in pairs],
                    "actual_weights": [float(v) for _, v in pairs],
                }
            )
    if not events:
        raise RuntimeError(
            f"No 'M1 RGB-D weights' lines found in {path}. "
            "Use the diagnostic config with m1_mapping_log_every: 1 and capture "
            "stdout/stderr with tee."
        )
    return events


def load_m1_signals(directory, signal):
    out = {
        0: {
            "u": 1.0,
            "confidence": 1.0,
            "sigma_t": float("nan"),
            "sigma_r": float("nan"),
            "valid": False,
        }
    }
    for path in sorted(directory.glob("*.pt")):
        d = torch.load(path, map_location="cpu")
        frame = int(d.get("frame", int(path.stem)))
        clusters = d.get("P_xi_clusters", {})
        P = clusters.get(signal.block_size)
        if P is None:
            P = clusters.get(str(signal.block_size))
        if not isinstance(P, torch.Tensor):
            continue
        try:
            s = signal.evaluate(P.double())
        except Exception:
            continue
        if all(finite(s[k]) for k in ("u", "confidence", "sigma_t", "sigma_r")):
            s["valid"] = True
            out[frame] = s
    return out


def fit_se3_alignment(est_xyz, gt_xyz):
    xbar = est_xyz.mean(axis=0)
    ybar = gt_xyz.mean(axis=0)
    X, Y = est_xyz - xbar, gt_xyz - ybar
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = ybar - R @ xbar
    A = np.eye(4, dtype=np.float64)
    A[:3, :3] = R
    A[:3, 3] = t
    return A


def scalar_value(v):
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            return None
        v = v.item()
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if finite(v) else None


def load_m2a(directory):
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
        obs = {}
        raw_obs = d.get("m2a3_observation_diag")
        if isinstance(raw_obs, dict):
            for key in OBS_KEYS:
                value = scalar_value(raw_obs.get(key))
                if value is not None:
                    obs[key] = value
        samples.append(
            {
                "frame": int(d.get("frame", int(path.stem))),
                "T": T,
                "G": G,
                "obs": obs,
            }
        )

    if len(samples) < 3:
        return {}, None

    est_xyz = np.stack([
        torch.linalg.inv(s["T"])[:3, 3].numpy() for s in samples
    ])
    gt_xyz = np.stack([
        torch.linalg.inv(s["G"])[:3, 3].numpy() for s in samples
    ])
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
            **s["obs"],
        }
    return out, A_np.tolist()


def enrich_events(events, m1, m2a, clip_min, clip_max):
    rows = []
    enriched = []
    for event in events:
        signals = []
        for frame in event["frames"]:
            signals.append(
                m1.get(
                    frame,
                    {
                        "u": 1.0,
                        "confidence": 1.0,
                        "sigma_t": float("nan"),
                        "sigma_r": float("nan"),
                        "valid": False,
                    },
                )
            )
        conf = [s["confidence"] for s in signals]
        raw_w = normalized_window_weights(conf, None, None)
        candidate_w = normalized_window_weights(conf, clip_min, clip_max)
        hit = [
            (w < clip_min if clip_min is not None else False)
            or (w > clip_max if clip_max is not None else False)
            for w in raw_w
        ]

        e = dict(event)
        e.update(
            {
                "u": [s["u"] for s in signals],
                "confidence": conf,
                "raw_normalized_weights": raw_w,
                "candidate_weights": candidate_w,
                "candidate_clip_hits": hit,
                "ess_actual": ess(event["actual_weights"]),
                "ess_raw": ess(raw_w),
                "ess_candidate": ess(candidate_w),
                "max_actual_weight": max(event["actual_weights"]),
                "max_candidate_weight": max(candidate_w),
                "min_candidate_weight": min(candidate_w),
                "std_candidate_weight": float(np.std(candidate_w)),
            }
        )
        enriched.append(e)

        for j, frame in enumerate(event["frames"]):
            row = {
                "map_call": event["map_call"],
                "log_line": event["log_line"],
                "frame": frame,
                "u": signals[j]["u"],
                "confidence": signals[j]["confidence"],
                "sigma_t": signals[j].get("sigma_t", float("nan")),
                "sigma_r": signals[j].get("sigma_r", float("nan")),
                "m1_valid": bool(signals[j].get("valid", False)),
                "actual_weight": event["actual_weights"][j],
                "raw_normalized_weight": raw_w[j],
                "candidate_weight": candidate_w[j],
                "candidate_clip_hit": hit[j],
                "ess_actual": e["ess_actual"],
                "ess_candidate": e["ess_candidate"],
            }
            row.update(m2a.get(frame, {}))
            rows.append(row)
    return enriched, rows


def correlation_report(rows):
    metrics = [
        "pose_center_error_m",
        "pose_log_translation_error_m",
        "pose_rotation_error_rad",
        *OBS_KEYS,
    ]
    out = {}
    for metric in metrics:
        usable = [r for r in rows if metric in r and finite(r[metric])]
        if len(usable) < 3:
            continue
        out[metric] = {
            "count_occurrences": len(usable),
            "spearman_u": spearman(
                [r["u"] for r in usable], [r[metric] for r in usable]
            ),
            "spearman_candidate_weight": spearman(
                [r["candidate_weight"] for r in usable],
                [r[metric] for r in usable],
            ),
        }
    return out


def summarize(events, rows):
    return {
        "mapping_calls": len(events),
        "view_occurrences": len(rows),
        "unique_frames": len(set(r["frame"] for r in rows)),
        "ess_actual": stats([e["ess_actual"] for e in events]),
        "ess_candidate": stats([e["ess_candidate"] for e in events]),
        "max_candidate_weight": stats([e["max_candidate_weight"] for e in events]),
        "min_candidate_weight": stats([e["min_candidate_weight"] for e in events]),
        "candidate_clip_fraction": float(
            np.mean([r["candidate_clip_hit"] for r in rows])
        ) if rows else float("nan"),
        "u": stats([r["u"] for r in rows]),
        "correlations_occurrence_weighted_descriptive": correlation_report(rows),
    }


def write_csv(path, run_name, rows):
    fields = [
        "run", "map_call", "log_line", "frame", "u", "confidence",
        "sigma_t", "sigma_r", "m1_valid", "actual_weight",
        "raw_normalized_weight", "candidate_weight", "candidate_clip_hit",
        "ess_actual", "ess_candidate", "pose_center_error_m",
        "pose_log_translation_error_m", "pose_rotation_error_rad", *OBS_KEYS,
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"run": run_name, **row})


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", nargs="+", required=True,
        metavar="NAME=LOG,M1_DIR,M2A_DIR",
        help="One or more diagnostic runs.",
    )
    p.add_argument("--signal-file", type=Path, required=True)
    p.add_argument("--signal-fold", required=True)
    p.add_argument("--clip-min", type=float, default=0.25)
    p.add_argument("--clip-max", type=float, default=4.0)
    p.add_argument(
        "--output", type=Path,
        default=Path("results/m1_mapping_failure_audit.json"),
    )
    args = p.parse_args()

    signal = M1MappingSignal(args.signal_file, args.signal_fold)
    report = {
        "method": "m1_mapping_failure_mechanism_audit",
        "signal_file": str(args.signal_file),
        "signal_fold": args.signal_fold,
        "candidate_clip_min": args.clip_min,
        "candidate_clip_max": args.clip_max,
        "runs": {},
        "guardrail": (
            "Candidate weights are reconstructed offline from saved M1 covariance. "
            "M2-A3 residuals are observation-consistency diagnostics, not the exact "
            "runtime mapping RGB-D loss. Correlations over repeated keyframe "
            "appearances are descriptive because observations are dependent."
        ),
    }

    csv_path = args.output.with_suffix(".csv")
    if csv_path.exists():
        csv_path.unlink()

    print("=" * 88)
    print("M1 mapping failure-mechanism audit")
    print("=" * 88)
    print(
        f"fold={args.signal_fold} block={signal.block_size} mode={signal.mode} "
        f"candidate_clip=[{args.clip_min}, {args.clip_max}]"
    )

    for spec in args.run:
        name, log_path, m1_dir, m2a_dir = parse_run_spec(spec)
        events = parse_weight_log(log_path)
        m1 = load_m1_signals(m1_dir, signal)
        m2a, alignment = load_m2a(m2a_dir)
        events, rows = enrich_events(
            events, m1, m2a, args.clip_min, args.clip_max
        )
        summary = summarize(events, rows)
        report["runs"][name] = {
            "log": str(log_path),
            "m1_dir": str(m1_dir),
            "m2a_dir": str(m2a_dir),
            "alignment_A_c2w_est_to_gt": alignment,
            "summary": summary,
            "events": events,
        }
        write_csv(csv_path, name, rows)

        low_ess = sorted(events, key=lambda e: e["ess_candidate"])[:5]
        print(f"\n[{name}]")
        print(
            f"  mapping calls={summary['mapping_calls']}  "
            f"view occurrences={summary['view_occurrences']}  "
            f"unique frames={summary['unique_frames']}"
        )
        s = summary["ess_candidate"]
        print(
            "  candidate ESS q05/median/q95: "
            f"{s['q05']:.3f} / {s['median']:.3f} / {s['q95']:.3f}"
        )
        w = summary["max_candidate_weight"]
        print(
            "  max candidate weight median/q95/max: "
            f"{w['median']:.3f} / {w['q95']:.3f} / {w['max']:.3f}"
        )
        print(
            "  candidate pre-clip hit fraction: "
            f"{summary['candidate_clip_fraction']:.3%}"
        )
        print("  five lowest-ESS windows:")
        for e in low_ess:
            print(
                f"    call={e['map_call']:5d} ESS={e['ess_candidate']:.3f} "
                f"frames={e['frames']} "
                f"w={[round(x, 3) for x in e['candidate_weights']]}"
            )

        corr = summary["correlations_occurrence_weighted_descriptive"]
        for metric in [
            "pose_center_error_m",
            "pose_rotation_error_rad",
            "rgb_abs_median",
            "depth_abs_median_m",
        ]:
            if metric in corr:
                c = corr[metric]
                print(
                    f"  Spearman u / candidate_w vs {metric}: "
                    f"{c['spearman_u']:.3f} / "
                    f"{c['spearman_candidate_weight']:.3f} "
                    f"(n_occ={c['count_occurrences']})"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    print(f"\nSaved JSON: {args.output}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
