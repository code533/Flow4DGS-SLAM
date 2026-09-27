#!/usr/bin/env python3
"""M2-A3 offline innovation-consistency audit.

This script does NOT change the SLAM runtime.  It consumes the per-frame
M2-A/M2-A2 diagnostics already written under m2a_pose_uncertainty/ and asks:

  1) How large is the tracking correction relative to the motion prior?
  2) Is that correction statistically compatible with the saved prior and
     tracking covariances?
  3) Does the resulting innovation score separate nominal sequences from a
     known failure/OOD sequence?

Conventions
-----------
The saved poses are world-to-camera transforms.  Absolute pose covariance is
right-invariant with twist order [rho, phi].

Innovation:
    nu = Log(T_prior^{-1} T_track)

For finite innovations, the covariance of nu is propagated with numerical
Jacobians of Log(T_prior^{-1} T_track) with respect to right perturbations of
both poses:
    S = J_prior P_prior J_prior^T + J_track P_track J_track^T

The sum assumes zero cross-covariance between prior and tracking estimates.
That independence assumption is not exact here because tracking is initialized
from the prior and both estimates share SLAM history/map information.  The
reported d^2 is therefore an innovation-consistency / Mahalanobis score, not a
claim of an exact chi-square NIS.  Chi-square cutoffs are printed only as
diagnostic references.

M2-A2 tracking calibration, when supplied, is the frozen whitened Diag-6:
    P_track_cal = P_track_raw^(1/2) C_diag P_track_raw^(1/2)

Example
-------
python scripts/audit_m2a3_innovation_multiseq.py \
  --sequence walking_xyz=results/tum/.../m2a_pose_uncertainty \
             walking_static=results/tum/.../m2a_pose_uncertainty \
             sitting_rpy=results/tum/.../m2a_pose_uncertainty \
             placing_box3=results/bonn/.../m2a_pose_uncertainty \
  --calibration-report results/m2a2_tracking_loso_2.json \
  --calibration-fold placing_box3 \
  --nominal walking_xyz walking_static sitting_rpy \
  --failure placing_box3 \
  --output results/m2a3_innovation_audit.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.m2_uncertainty import SE3_log
from utils.pose_utils import SE3_exp


CHI2_6_95 = 12.591587
CHI2_6_99 = 16.811894


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


def sym(P):
    return 0.5 * (P + P.T)


def finite_matrix(x, shape):
    return (
        isinstance(x, torch.Tensor)
        and tuple(x.shape) == tuple(shape)
        and bool(torch.isfinite(x).all())
    )


def stable_psd_sqrt(P, floor_rel=1e-10):
    P = sym(P)
    eig, vec = torch.linalg.eigh(P)
    mx = eig.max().clamp_min(1e-18)
    eig = eig.clamp_min(mx * float(floor_rel))
    root = (vec * torch.sqrt(eig).unsqueeze(0)) @ vec.T
    return sym(root)


def apply_diag6(P, diag, floor_rel=1e-10):
    root = stable_psd_sqrt(P, floor_rel)
    d = torch.as_tensor(diag, dtype=P.dtype, device=P.device).reshape(6)
    if not bool(torch.isfinite(d).all()) or bool((d <= 0).any()):
        raise ValueError("Diag-6 calibration entries must be finite and positive")
    return sym(root @ torch.diag(d) @ root)


def load_diag6_report(path, fold=None):
    path = Path(path).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))

    if "folds" in data:
        if fold is None:
            raise ValueError(
                "Calibration report contains LOSO folds; "
                "--calibration-fold is required."
            )
        if fold not in data["folds"]:
            raise KeyError(
                f"Fold {fold!r} not found. "
                f"Available: {sorted(data['folds'].keys())}"
            )
        node = data["folds"][fold]
        diag = node["calibration"]["diag6"]["diag"]
        train_sequences = list(node.get("train_sequences", []))
        source = f"{path}:fold={fold}"
    elif "calibration" in data and "diag6" in data["calibration"]:
        diag = data["calibration"]["diag6"]["diag"]
        train_sequences = list(data.get("train_sequences", []))
        source = str(path)
    else:
        raise ValueError(
            "Unsupported M2-A2 calibration report layout. Expected "
            "folds/<fold>/calibration/diag6/diag."
        )

    d = torch.as_tensor(diag, dtype=torch.float64).reshape(-1)
    if d.numel() != 6:
        raise ValueError(f"Expected 6 Diag-6 entries, got {d.numel()}")
    if not bool(torch.isfinite(d).all()) or bool((d <= 0).any()):
        raise ValueError("Diag-6 entries must be finite and positive")

    return {
        "diag": d.tolist(),
        "sigma_multipliers": torch.sqrt(d).tolist(),
        "source": source,
        "train_sequences": train_sequences,
        "held_out": fold,
    }


def innovation(T_prior, T_track):
    return SE3_log(torch.linalg.inv(T_prior) @ T_track)


def innovation_jacobians(T_prior, T_track, eps):
    """Finite-difference Jacobians for right perturbations of both poses."""
    Jp = torch.empty((6, 6), dtype=torch.float64)
    Jt = torch.empty((6, 6), dtype=torch.float64)

    for j in range(6):
        e = torch.zeros(6, dtype=torch.float64)
        e[j] = float(eps)
        Ep = SE3_exp(e)
        Em = SE3_exp(-e)

        rp = innovation(T_prior @ Ep, T_track)
        rm = innovation(T_prior @ Em, T_track)
        Jp[:, j] = (rp - rm) / (2.0 * float(eps))

        rp = innovation(T_prior, T_track @ Ep)
        rm = innovation(T_prior, T_track @ Em)
        Jt[:, j] = (rp - rm) / (2.0 * float(eps))

    return Jp, Jt


def stable_quad(x, P, floor_rel=1e-10):
    P = sym(P)
    eig, vec = torch.linalg.eigh(P)
    mx = eig.max().clamp_min(1e-18)
    eig = eig.clamp_min(mx * float(floor_rel))
    y = vec.T @ x
    return float(((y * y) / eig).sum())


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


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def quantile(values, q):
    x = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(x, float(q)))


def roc_auc(scores, labels):
    """Rank-based ROC AUC; labels are 0=nominal, 1=failure."""
    if len(scores) != len(labels) or not scores:
        return float("nan")
    labels = torch.tensor(labels, dtype=torch.int64)
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    if n0 == 0 or n1 == 0:
        return float("nan")
    ranks = torch.tensor(rankdata(scores), dtype=torch.float64)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def load_sequence(name, directory, diag, fd_eps, eig_floor_rel):
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {directory}")

    samples = []
    skipped = {}
    source_counts = {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for path in files:
        d = torch.load(path, map_location="cpu")
        required = ("T_motion_prior", "T_final", "P_abs_prior_right")
        if any(d.get(k, None) is None for k in required):
            skip("missing_prior_or_pose")
            continue

        P_track = d.get("P_track_right_raw", None)
        source = "P_track_right_raw"
        if P_track is None:
            P_track = d.get("P_track_right", None)
            source = "P_track_right"
        if P_track is None:
            skip("missing_tracking_covariance")
            continue

        T_prior = d["T_motion_prior"].double()
        T_track = d["T_final"].double()
        P_prior = sym(d["P_abs_prior_right"].double())
        P_track = sym(P_track.double())

        if not (
            finite_matrix(T_prior, (4, 4))
            and finite_matrix(T_track, (4, 4))
            and finite_matrix(P_prior, (6, 6))
            and finite_matrix(P_track, (6, 6))
        ):
            skip("nonfinite_or_bad_shape")
            continue

        nu = innovation(T_prior, T_track)
        if not bool(torch.isfinite(nu).all()):
            skip("nonfinite_innovation")
            continue

        Jp, Jt = innovation_jacobians(T_prior, T_track, fd_eps)
        if not bool(torch.isfinite(Jp).all() and torch.isfinite(Jt).all()):
            skip("nonfinite_jacobian")
            continue

        P_track_cal = apply_diag6(P_track, diag, eig_floor_rel)

        S_raw = sym(Jp @ P_prior @ Jp.T + Jt @ P_track @ Jt.T)
        S_cal = sym(Jp @ P_prior @ Jp.T + Jt @ P_track_cal @ Jt.T)

        # First-order near-identity reference.  This is useful for checking
        # whether finite innovation geometry materially changes the score.
        S_naive_cal = sym(P_prior + P_track_cal)

        item = {
            "sequence": name,
            "frame": int(d.get("frame", -1)),
            "source": source,
            "nu_t": float(torch.linalg.norm(nu[:3])),
            "nu_r": float(torch.linalg.norm(nu[3:])),
            "d2_raw": stable_quad(nu, S_raw, eig_floor_rel),
            "d2_cal": stable_quad(nu, S_cal, eig_floor_rel),
            "d2_naive_cal": stable_quad(nu, S_naive_cal, eig_floor_rel),
            "jp_rms": float(torch.sqrt(torch.mean(Jp * Jp))),
            "jt_rms": float(torch.sqrt(torch.mean(Jt * Jt))),
        }

        if d.get("T_gt", None) is not None:
            T_gt = d["T_gt"].double()
            if finite_matrix(T_gt, (4, 4)):
                e = SE3_log(torch.linalg.inv(T_track) @ T_gt)
                if bool(torch.isfinite(e).all()):
                    item["gt_error_t"] = float(torch.linalg.norm(e[:3]))
                    item["gt_error_r"] = float(torch.linalg.norm(e[3:]))

        samples.append(item)
        source_counts[source] = source_counts.get(source, 0) + 1

    if not samples:
        raise RuntimeError(
            f"No usable M2-A3 diagnostics in {directory}. "
            "Need T_motion_prior, T_final, P_abs_prior_right and tracking covariance."
        )

    return samples, source_counts, skipped


def summarize(samples):
    d2 = [s["d2_cal"] for s in samples]
    d2raw = [s["d2_raw"] for s in samples]
    d2naive = [s["d2_naive_cal"] for s in samples]
    nut = [s["nu_t"] for s in samples]
    nur = [s["nu_r"] for s in samples]

    out = {
        "count": len(samples),
        "innovation_t_median": quantile(nut, 0.5),
        "innovation_t_p95": quantile(nut, 0.95),
        "innovation_r_median": quantile(nur, 0.5),
        "innovation_r_p95": quantile(nur, 0.95),
        "d2_raw_median": quantile(d2raw, 0.5),
        "d2_raw_p95": quantile(d2raw, 0.95),
        "d2_cal_median": quantile(d2, 0.5),
        "d2_cal_p95": quantile(d2, 0.95),
        "d2_cal_p99": quantile(d2, 0.99),
        "d2_cal_max": max(d2),
        "reference_chi2_6_95_coverage": sum(x <= CHI2_6_95 for x in d2) / len(d2),
        "reference_chi2_6_99_coverage": sum(x <= CHI2_6_99 for x in d2) / len(d2),
        "finite_geometry_ratio_median": quantile(
            [a / max(b, 1e-30) for a, b in zip(d2, d2naive)], 0.5
        ),
    }

    gt = [s for s in samples if "gt_error_t" in s and "gt_error_r" in s]
    if gt:
        et = [s["gt_error_t"] for s in gt]
        er = [s["gt_error_r"] for s in gt]
        score = [s["d2_cal"] for s in gt]
        out.update({
            "gt_count": len(gt),
            "gt_error_t_median": quantile(et, 0.5),
            "gt_error_r_median": quantile(er, 0.5),
            "d2_vs_gt_translation_pearson": pearson(score, et),
            "d2_vs_gt_translation_spearman": spearman(score, et),
            "d2_vs_gt_rotation_pearson": pearson(score, er),
            "d2_vs_gt_rotation_spearman": spearman(score, er),
        })
    return out


def nominal_failure_analysis(samples_by_name, nominal, failure):
    missing = [n for n in nominal + failure if n not in samples_by_name]
    if missing:
        raise KeyError(f"Unknown nominal/failure sequence names: {missing}")

    nom = [s for n in nominal for s in samples_by_name[n]]
    fail = [s for n in failure for s in samples_by_name[n]]
    nom_scores = [s["d2_cal"] for s in nom]
    fail_scores = [s["d2_cal"] for s in fail]

    thresholds = {
        "nominal_q95": quantile(nom_scores, 0.95),
        "nominal_q99": quantile(nom_scores, 0.99),
        "nominal_q999": quantile(nom_scores, 0.999),
    }
    exceedance = {}
    for key, thr in thresholds.items():
        exceedance[key] = {
            "threshold": thr,
            "nominal_exceedance": sum(x > thr for x in nom_scores) / len(nom_scores),
            "failure_exceedance": sum(x > thr for x in fail_scores) / len(fail_scores),
        }

    scores = nom_scores + fail_scores
    labels = [0] * len(nom_scores) + [1] * len(fail_scores)
    return {
        "nominal_sequences": nominal,
        "failure_sequences": failure,
        "nominal_frames": len(nom_scores),
        "failure_frames": len(fail_scores),
        "sequence_label_roc_auc": roc_auc(scores, labels),
        "threshold_analysis": exceedance,
    }


def print_summary(name, m):
    print(f"\n{name}")
    print(f"  frames: {m['count']}")
    print(
        "  innovation median [t/r]: "
        f"{m['innovation_t_median']:.6g} m / "
        f"{m['innovation_r_median']:.6g} rad"
    )
    print(
        "  d2 calibrated: "
        f"median={m['d2_cal_median']:.6g}, "
        f"p95={m['d2_cal_p95']:.6g}, "
        f"p99={m['d2_cal_p99']:.6g}, "
        f"max={m['d2_cal_max']:.6g}"
    )
    print(
        "  chi2(6) reference coverage [95/99]: "
        f"{100*m['reference_chi2_6_95_coverage']:.2f}% / "
        f"{100*m['reference_chi2_6_99_coverage']:.2f}%"
    )
    if "gt_count" in m:
        print(
            "  GT median error [t/r]: "
            f"{m['gt_error_t_median']:.6g} m / "
            f"{m['gt_error_r_median']:.6g} rad"
        )
        print(
            "  d2 vs GT Spearman [t/r]: "
            f"{m['d2_vs_gt_translation_spearman']:.4f} / "
            f"{m['d2_vs_gt_rotation_spearman']:.4f}"
        )


def main():
    p = argparse.ArgumentParser(
        description="M2-A3 innovation/failure-separation audit"
    )
    p.add_argument(
        "--sequence", nargs="+", required=True, metavar="NAME=PATH",
        help="Per-sequence m2a_pose_uncertainty directories.",
    )
    p.add_argument(
        "--calibration-report", type=Path, default=None,
        help="M2-A2 calibration JSON. If omitted, identity Diag-6 is used.",
    )
    p.add_argument(
        "--calibration-fold", default=None,
        help="LOSO held-out fold whose Diag-6 is frozen for this audit.",
    )
    p.add_argument(
        "--nominal", nargs="*", default=[],
        help="Sequence names treated as nominal for separation analysis.",
    )
    p.add_argument(
        "--failure", nargs="*", default=[],
        help="Sequence names treated as failure/OOD for separation analysis.",
    )
    p.add_argument("--fd-eps", type=float, default=1e-6)
    p.add_argument("--eig-floor-rel", type=float, default=1e-10)
    p.add_argument(
        "--output", type=Path,
        default=Path("results/m2a3_innovation_audit.json"),
    )
    args = p.parse_args()

    specs = parse_specs(args.sequence)

    if args.calibration_report is None:
        calibration = {
            "diag": [1.0] * 6,
            "sigma_multipliers": [1.0] * 6,
            "source": "identity/no-calibration",
            "train_sequences": [],
            "held_out": None,
        }
    else:
        calibration = load_diag6_report(
            args.calibration_report, args.calibration_fold
        )

    print("=" * 78)
    print("M2-A3 innovation-consistency audit")
    print("=" * 78)
    print("calibration:", calibration["source"])
    print("Diag-6:", [round(x, 6) for x in calibration["diag"]])
    print(
        "sigma multipliers:",
        [round(x, 6) for x in calibration["sigma_multipliers"]],
    )
    print(
        "NOTE: d2 uses zero prior/track cross-covariance; "
        "treat chi-square thresholds as references, not exact NIS guarantees."
    )

    samples_by_name = {}
    paths = {}
    sources = {}
    skipped = {}
    summaries = {}

    for name, path in specs:
        xs, src, sk = load_sequence(
            name, path, calibration["diag"],
            args.fd_eps, args.eig_floor_rel,
        )
        samples_by_name[name] = xs
        paths[name] = str(path)
        sources[name] = src
        skipped[name] = sk
        summaries[name] = summarize(xs)
        print_summary(name, summaries[name])

    separation = None
    if args.nominal or args.failure:
        if not args.nominal or not args.failure:
            raise ValueError(
                "--nominal and --failure must be supplied together."
            )
        separation = nominal_failure_analysis(
            samples_by_name, args.nominal, args.failure
        )
        print("\n" + "-" * 78)
        print("Nominal vs failure/OOD separation")
        print(
            "  sequence-label ROC AUC: "
            f"{separation['sequence_label_roc_auc']:.6f}"
        )
        for key, node in separation["threshold_analysis"].items():
            print(
                f"  {key}: threshold={node['threshold']:.6g}, "
                f"nominal>{100*node['nominal_exceedance']:.2f}%, "
                f"failure>{100*node['failure_exceedance']:.2f}%"
            )

    report = {
        "model": "m2a3_innovation_consistency",
        "sequence_paths": paths,
        "covariance_sources": sources,
        "skipped": skipped,
        "calibration": calibration,
        "fd_eps": args.fd_eps,
        "eig_floor_rel": args.eig_floor_rel,
        "independence_assumption": (
            "S=J_prior P_prior J_prior^T + J_track P_track J_track^T; "
            "prior/track cross-covariance is not modeled."
        ),
        "per_sequence": summaries,
        "nominal_failure_analysis": separation,
        "per_frame": samples_by_name,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
