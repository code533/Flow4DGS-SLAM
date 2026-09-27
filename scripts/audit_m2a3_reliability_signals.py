#!/usr/bin/env python3
"""M2-A3 offline audit of failure-aware reliability signals.

This script is intentionally diagnostic-only: it does not modify SLAM runtime
or posterior fusion.  It joins the already-saved M1 and M2-A2 per-frame
diagnostics and tests whether any *internal* signal tracks GT pose error and/or
separates nominal sequences from a held-out failure/OOD sequence.

The purpose is feature discovery, not classifier fitting.  Every candidate is
reported individually; no learned combination is fit on the failure sequence.

Input
-----
--sequence NAME=.../m2a_pose_uncertainty

The sibling directory .../m1_pose_uncertainty is used automatically when it
exists.

Example
-------
python scripts/audit_m2a3_reliability_signals.py \
  --sequence walking_xyz=results/tum/.../m2a_pose_uncertainty \
             walking_static=results/tum/.../m2a_pose_uncertainty \
             sitting_rpy=results/tum/.../m2a_pose_uncertainty \
             placing_box3=results/bonn/.../m2a_pose_uncertainty \
  --nominal walking_xyz walking_static sitting_rpy \
  --failure placing_box3 \
  --output results/m2a3_reliability_signals.json
"""

import argparse
import json
import math
from pathlib import Path

import torch

from utils.m2_uncertainty import SE3_log


def parse_specs(values):
    out = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty sequence name in {value!r}")
        out.append((name, Path(path).expanduser()))
    if len({n for n, _ in out}) != len(out):
        raise ValueError("Sequence names must be unique.")
    return out


def scalar(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            return None
        x = float(x.detach().cpu().reshape(()))
    else:
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
    return x if math.isfinite(x) else None


def positive_log10(x, floor=1e-30):
    x = scalar(x)
    if x is None or x <= 0:
        return None
    return math.log10(max(x, floor))


def matrix_stats(P, prefix):
    out = {}
    if not isinstance(P, torch.Tensor) or tuple(P.shape) != (6, 6):
        return out
    P = 0.5 * (P.double() + P.double().T)
    if not bool(torch.isfinite(P).all()):
        return out
    eig = torch.linalg.eigvalsh(P)
    mx = eig.max().clamp_min(1e-30)
    eig = eig.clamp_min(mx * 1e-10)
    out[f"{prefix}_log10_trace"] = math.log10(
        max(float(torch.trace(P)), 1e-30)
    )
    out[f"{prefix}_log10_det"] = float(torch.log10(eig).sum())
    out[f"{prefix}_sigma_t_rms"] = float(
        torch.sqrt(torch.diagonal(P)[:3].clamp_min(0).mean())
    )
    out[f"{prefix}_sigma_r_rms"] = float(
        torch.sqrt(torch.diagonal(P)[3:].clamp_min(0).mean())
    )
    return out


def rankdata(values):
    x = torch.tensor(values, dtype=torch.float64)
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks.tolist()


def pearson(x, y):
    if len(x) < 2:
        return float("nan")
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum())
    if float(den) <= 0:
        return float("nan")
    return float((x * y).sum() / den)


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def quantile(values, q):
    if not values:
        return float("nan")
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q))


def roc_auc(scores, labels):
    if len(scores) != len(labels) or not scores:
        return float("nan")
    lab = torch.tensor(labels, dtype=torch.int64)
    n1 = int((lab == 1).sum())
    n0 = int((lab == 0).sum())
    if n0 == 0 or n1 == 0:
        return float("nan")
    ranks = torch.tensor(rankdata(scores), dtype=torch.float64)
    rank_sum_pos = float(ranks[lab == 1].sum())
    return (rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def load_pt_by_frame(directory):
    if not directory.exists():
        return {}
    out = {}
    for path in sorted(directory.glob("*.pt")):
        d = torch.load(path, map_location="cpu")
        frame = int(d.get("frame", int(path.stem)))
        out[frame] = d
    return out


def extract_m2_features(d):
    f = {}

    direct = {
        "m2_rgb_scale": d.get("m2a2_rgb_scale"),
        "m2_depth_scale": d.get("m2a2_depth_scale"),
        "m2_num_pixels": d.get("m2a2_num_pixels"),
        "m2_num_observations": d.get("m2a2_num_observations"),
        "m2_tracking_correction_t": d.get(
            "tracking_correction_translation_m"
        ),
        "m2_tracking_correction_r": d.get(
            "tracking_correction_rotation_rad"
        ),
    }
    for k, v in direct.items():
        z = scalar(v)
        if z is not None:
            f[k] = z

    log_direct = {
        "m2_log10_condition": d.get("m2a2_condition"),
        "m2_log10_bread_condition": d.get("m2a2_bread_condition"),
        "m2_log10_bread_trace": d.get("m2a2_bread_trace"),
    }
    for k, v in log_direct.items():
        z = positive_log10(v)
        if z is not None:
            f[k] = z

    f.update(matrix_stats(d.get("P_track_right_raw"), "m2_track_cov"))
    f.update(matrix_stats(d.get("P_abs_prior_right"), "m2_prior_cov"))
    return f


def extract_m1_features(d):
    f = {}
    direct = {
        "m1_fb_error_median_px": d.get("fb_error_median_px"),
        "m1_maha_median": d.get("maha_median"),
        "m1_valid_extent": d.get("valid_extent"),
        "m1_num_pixels": d.get("num_pixels"),
        "m1_motion_scale_joint": d.get("motion_scale_joint"),
        "m1_motion_scale_translation": d.get(
            "motion_scale_translation"
        ),
    }
    for k, v in direct.items():
        z = scalar(v)
        if z is not None:
            f[k] = z

    for k, v in {
        "m1_log10_condition": d.get("condition"),
        "m1_log10_kappa": d.get("kappa"),
    }.items():
        z = positive_log10(v)
        if z is not None:
            f[k] = z

    f.update(matrix_stats(d.get("P_xi_cluster"), "m1_cluster_cov"))
    return f


def load_sequence(name, m2_dir):
    m2 = load_pt_by_frame(m2_dir)
    if not m2:
        raise FileNotFoundError(f"No .pt files in {m2_dir}")

    m1_dir = m2_dir.parent / "m1_pose_uncertainty"
    m1 = load_pt_by_frame(m1_dir)

    rows = []
    for frame, d in sorted(m2.items()):
        T_final = d.get("T_final")
        T_gt = d.get("T_gt")
        if not (
            isinstance(T_final, torch.Tensor)
            and isinstance(T_gt, torch.Tensor)
            and tuple(T_final.shape) == (4, 4)
            and tuple(T_gt.shape) == (4, 4)
        ):
            continue

        T_final = T_final.double()
        T_gt = T_gt.double()
        if not bool(torch.isfinite(T_final).all() and torch.isfinite(T_gt).all()):
            continue

        e = SE3_log(torch.linalg.inv(T_final) @ T_gt)
        if not bool(torch.isfinite(e).all()):
            continue

        features = extract_m2_features(d)
        if frame in m1:
            features.update(extract_m1_features(m1[frame]))

        rows.append({
            "sequence": name,
            "frame": frame,
            "gt_error_t": float(torch.linalg.norm(e[:3])),
            "gt_error_r": float(torch.linalg.norm(e[3:])),
            "features": features,
        })

    if not rows:
        raise RuntimeError(f"No usable GT-linked frames in {m2_dir}")
    return rows, str(m1_dir), bool(m1)


def feature_names(rows_by_name):
    names = set()
    for rows in rows_by_name.values():
        for r in rows:
            names.update(r["features"].keys())
    return sorted(names)


def summarize_feature(rows, feature):
    xs, et, er = [], [], []
    for r in rows:
        if feature in r["features"]:
            xs.append(r["features"][feature])
            et.append(r["gt_error_t"])
            er.append(r["gt_error_r"])
    if not xs:
        return None
    return {
        "count": len(xs),
        "median": quantile(xs, 0.5),
        "q05": quantile(xs, 0.05),
        "q95": quantile(xs, 0.95),
        "spearman_gt_translation": spearman(xs, et),
        "spearman_gt_rotation": spearman(xs, er),
    }


def separation(rows_by_name, feature, nominal, failure):
    nom = [
        r["features"][feature]
        for n in nominal
        for r in rows_by_name[n]
        if feature in r["features"]
    ]
    fail = [
        r["features"][feature]
        for n in failure
        for r in rows_by_name[n]
        if feature in r["features"]
    ]
    if not nom or not fail:
        return None

    scores = nom + fail
    labels = [0] * len(nom) + [1] * len(fail)
    auc_high = roc_auc(scores, labels)
    auc_low = 1.0 - auc_high

    q05 = quantile(nom, 0.05)
    q95 = quantile(nom, 0.95)
    q01 = quantile(nom, 0.01)
    q99 = quantile(nom, 0.99)

    return {
        "nominal_count": len(nom),
        "failure_count": len(fail),
        "nominal_median": quantile(nom, 0.5),
        "failure_median": quantile(fail, 0.5),
        "auc_high_means_failure": auc_high,
        "auc_low_means_failure": auc_low,
        "nominal_q05": q05,
        "nominal_q95": q95,
        "nominal_q01": q01,
        "nominal_q99": q99,
        "failure_above_nominal_q95": sum(x > q95 for x in fail) / len(fail),
        "failure_below_nominal_q05": sum(x < q05 for x in fail) / len(fail),
        "failure_above_nominal_q99": sum(x > q99 for x in fail) / len(fail),
        "failure_below_nominal_q01": sum(x < q01 for x in fail) / len(fail),
    }


def main():
    p = argparse.ArgumentParser(
        description="M2-A3 offline reliability-signal audit"
    )
    p.add_argument(
        "--sequence", nargs="+", required=True, metavar="NAME=M2_DIR"
    )
    p.add_argument("--nominal", nargs="+", required=True)
    p.add_argument("--failure", nargs="+", required=True)
    p.add_argument(
        "--output", type=Path,
        default=Path("results/m2a3_reliability_signals.json"),
    )
    args = p.parse_args()

    specs = parse_specs(args.sequence)
    names = {n for n, _ in specs}
    missing = [
        n for n in args.nominal + args.failure if n not in names
    ]
    if missing:
        raise KeyError(f"Unknown sequence names: {missing}")

    rows_by_name = {}
    m1_paths = {}
    m1_available = {}
    for name, path in specs:
        rows, m1_path, has_m1 = load_sequence(name, path)
        rows_by_name[name] = rows
        m1_paths[name] = m1_path
        m1_available[name] = has_m1

    features = feature_names(rows_by_name)
    per_feature = {}

    print("=" * 86)
    print("M2-A3 failure-aware reliability signal audit")
    print("=" * 86)
    print(
        "Exploratory only: failure/OOD frames are used for evaluation, "
        "not for fitting a combined classifier."
    )
    print("M1 sibling diagnostics available:", m1_available)

    for feature in features:
        node = {
            "per_sequence": {},
            "nominal_failure": separation(
                rows_by_name, feature, args.nominal, args.failure
            ),
        }
        for name, rows in rows_by_name.items():
            node["per_sequence"][name] = summarize_feature(rows, feature)
        per_feature[feature] = node

    sortable = []
    for feature, node in per_feature.items():
        sep = node["nominal_failure"]
        if sep is None:
            continue
        strength = max(
            sep["auc_high_means_failure"],
            sep["auc_low_means_failure"],
        )
        direction = (
            "HIGH"
            if sep["auc_high_means_failure"] >=
               sep["auc_low_means_failure"]
            else "LOW"
        )
        sortable.append((strength, feature, direction, sep))

    sortable.sort(reverse=True)

    print("\nTop univariate sequence-label separation signals")
    print(
        "  (AUC is frame-level and temporally correlated; use only as an "
        "exploratory separation statistic.)"
    )
    for strength, feature, direction, sep in sortable[:12]:
        print(
            f"  {feature:32s} "
            f"AUC={strength:.4f} ({direction}=failure), "
            f"nom_med={sep['nominal_median']:.6g}, "
            f"fail_med={sep['failure_median']:.6g}, "
            f"fail>q95={100*sep['failure_above_nominal_q95']:.1f}%, "
            f"fail<q05={100*sep['failure_below_nominal_q05']:.1f}%"
        )

    print("\nWithin-sequence Spearman vs GT pose error")
    for feature in features:
        parts = []
        for name in rows_by_name:
            s = per_feature[feature]["per_sequence"][name]
            if s is None:
                continue
            parts.append(
                f"{name}: t={s['spearman_gt_translation']:.3f},"
                f" r={s['spearman_gt_rotation']:.3f}"
            )
        print(f"  {feature:32s} " + " | ".join(parts))

    report = {
        "model": "m2a3_reliability_signal_audit",
        "sequence_paths": {n: str(p) for n, p in specs},
        "m1_paths": m1_paths,
        "m1_available": m1_available,
        "nominal_sequences": args.nominal,
        "failure_sequences": args.failure,
        "warning": (
            "Frame-level ROC AUC is exploratory because frames are temporally "
            "correlated and sequence identity is confounded with domain. "
            "No combined classifier is fit in this audit."
        ),
        "features": per_feature,
        "per_frame": rows_by_name,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
