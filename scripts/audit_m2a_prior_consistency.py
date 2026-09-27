#!/usr/bin/env python3
"""Same-object consistency audit for the M2-A motion-prior covariance.

This script evaluates the covariance at the pose mean it actually belongs to:

    e_k^- = Log((T_motion_prior)^-1 T_gt)
    P_k^- = P_abs_prior_right

The repository uses world-to-camera poses and a right perturbation

    T_cw = Tbar_cw Exp(delta_xi^),

so e_k^- is the matching right-tangent GT error.

This is a consistency diagnostic, not a formal iid calibration test: frames
from one SLAM trajectory are temporally correlated, and a single trajectory
does not provide repeated draws of the estimator.

Example
-------
python scripts/audit_m2a_prior_consistency.py \
  --sequence placing_box=/run/box1/m2a_pose_uncertainty \
             placing_box2=/run/box2/m2a_pose_uncertainty \
             placing_box3=/run/box3/m2a_pose_uncertainty \
  --output results/m2a_prior_bonn_consistency.json
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


CHI2_3_95 = 7.814728
CHI2_3_99 = 11.344867
CHI2_6_95 = 12.591587
CHI2_6_99 = 16.811894


def parse_specs(values):
    out = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty sequence name in {value!r}")
        out.append((name, Path(raw_path).expanduser()))
    return out


def sym(P):
    return 0.5 * (P + P.T)


def stable_nees(e, P, floor_rel=1e-10):
    P = sym(P)
    eig, vec = torch.linalg.eigh(P)
    mx = eig.max().clamp_min(1e-18)
    eig = eig.clamp_min(mx * float(floor_rel))
    inv = (vec * (1.0 / eig).unsqueeze(0)) @ vec.T
    return float(e @ inv @ e)


def whiten(e, P, floor_rel=1e-10):
    P = sym(P)
    eig, vec = torch.linalg.eigh(P)
    mx = eig.max().clamp_min(1e-18)
    eig = eig.clamp_min(mx * float(floor_rel))
    invroot = (vec * (1.0 / torch.sqrt(eig)).unsqueeze(0)) @ vec.T
    return invroot @ e


def rms_sigma(P, sl):
    d = torch.diagonal(sym(P))[sl].clamp_min(0.0)
    return float(torch.sqrt(d.mean()))


def pearson(x, y):
    if len(x) < 2:
        return float("nan")
    a = torch.tensor(x, dtype=torch.float64)
    b = torch.tensor(y, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = torch.sqrt((a * a).sum() * (b * b).sum())
    if float(den) <= 0:
        return float("nan")
    return float((a * b).sum() / den)


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


def load_sequence(name, directory):
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {directory}")

    rows = []
    skipped = {
        "missing_target": 0,
        "nonfinite": 0,
    }
    for path in files:
        d = torch.load(path, map_location="cpu")
        T_prior = d.get("T_motion_prior", None)
        T_gt = d.get("T_gt", None)
        P_prior = d.get("P_abs_prior_right", None)
        if T_prior is None or T_gt is None or P_prior is None:
            skipped["missing_target"] += 1
            continue

        T_prior = T_prior.double()
        T_gt = T_gt.double()
        P_prior = sym(P_prior.double())
        if not (
            bool(torch.isfinite(T_prior).all())
            and bool(torch.isfinite(T_gt).all())
            and bool(torch.isfinite(P_prior).all())
        ):
            skipped["nonfinite"] += 1
            continue

        # Matching right-tangent error for
        # T_true = T_prior Exp(e^).
        e = SE3_log(torch.linalg.inv(T_prior) @ T_gt)

        row = {
            "sequence": name,
            "frame": int(d["frame"]),
            "error": e,
            "P": P_prior,
        }

        # P_abs_right_raw is retained as a secondary historical diagnostic.
        # It is not the selected M2-A prior because the selected chain may
        # include the previous M2-A2 posterior.
        P_raw = d.get("P_abs_right_raw", None)
        if P_raw is not None:
            P_raw = sym(P_raw.double())
            if bool(torch.isfinite(P_raw).all()):
                row["P_raw_chain"] = P_raw

        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No same-object M2-A prior diagnostics found in {directory}"
        )
    return rows, skipped


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def summarize(rows, covariance_key="P"):
    valid = [r for r in rows if covariance_key in r]
    if not valid:
        return None

    nees6, neest, neesr = [], [], []
    errt, errr, sigt, sigr = [], [], [], []
    z = []
    nonsymmetric = nonpsd = 0

    for r in valid:
        P = r[covariance_key]
        e = r["error"]

        if float(torch.max(torch.abs(P - P.T))) > 1e-8:
            nonsymmetric += 1
        eig = torch.linalg.eigvalsh(sym(P))
        if float(eig.min()) < -1e-10:
            nonpsd += 1

        nees6.append(stable_nees(e, P))
        neest.append(stable_nees(e[:3], P[:3, :3]))
        neesr.append(stable_nees(e[3:], P[3:, 3:]))
        errt.append(float(torch.linalg.norm(e[:3])))
        errr.append(float(torch.linalg.norm(e[3:])))
        sigt.append(rms_sigma(P, slice(0, 3)))
        sigr.append(rms_sigma(P, slice(3, 6)))
        z.append(whiten(e, P))

    n6 = torch.tensor(nees6, dtype=torch.float64)
    nt = torch.tensor(neest, dtype=torch.float64)
    nr = torch.tensor(neesr, dtype=torch.float64)
    et = torch.tensor(errt, dtype=torch.float64)
    er = torch.tensor(errr, dtype=torch.float64)
    st = torch.tensor(sigt, dtype=torch.float64)
    sr = torch.tensor(sigr, dtype=torch.float64)
    Z = torch.stack(z)

    return {
        "frames": len(valid),
        "numerical": {
            "nonsymmetric": nonsymmetric,
            "nonpsd": nonpsd,
        },
        "error": {
            "translation_median_m": float(et.median()),
            "translation_mean_m": float(et.mean()),
            "rotation_median_rad": float(er.median()),
            "rotation_mean_rad": float(er.mean()),
        },
        "sigma_rms": {
            "translation_median_m": float(st.median()),
            "rotation_median_rad": float(sr.median()),
        },
        "nees": {
            "mean_6d": float(n6.mean()),
            "median_6d": float(n6.median()),
            "mean_translation_3d": float(nt.mean()),
            "mean_rotation_3d": float(nr.mean()),
            "coverage_6d_95": float((n6 <= CHI2_6_95).double().mean()),
            "coverage_6d_99": float((n6 <= CHI2_6_99).double().mean()),
            "coverage_translation_95": float((nt <= CHI2_3_95).double().mean()),
            "coverage_translation_99": float((nt <= CHI2_3_99).double().mean()),
            "coverage_rotation_95": float((nr <= CHI2_3_95).double().mean()),
            "coverage_rotation_99": float((nr <= CHI2_3_99).double().mean()),
        },
        "error_uncertainty_correlation": {
            "translation_pearson": finite_or_none(pearson(sigt, errt)),
            "translation_spearman": finite_or_none(spearman(sigt, errt)),
            "rotation_pearson": finite_or_none(pearson(sigr, errr)),
            "rotation_spearman": finite_or_none(spearman(sigr, errr)),
        },
        "whitened_error": {
            "mean": [float(v) for v in Z.mean(dim=0)],
            "std": [float(v) for v in Z.std(dim=0, unbiased=False)],
        },
    }


def print_summary(name, result, skipped):
    print(f"\n=== {name}: selected M2-A motion prior ===")
    print(
        f"frames={result['frames']} "
        f"skipped_missing={skipped['missing_target']} "
        f"skipped_nonfinite={skipped['nonfinite']}"
    )
    print(
        "prior GT error median: "
        f"t={result['error']['translation_median_m']:.6g} m, "
        f"r={result['error']['rotation_median_rad']:.6g} rad"
    )
    print(
        "prior sigma RMS median: "
        f"t={result['sigma_rms']['translation_median_m']:.6g} m, "
        f"r={result['sigma_rms']['rotation_median_rad']:.6g} rad"
    )
    n = result["nees"]
    print(
        "NEES 6D: "
        f"mean={n['mean_6d']:.6g}, median={n['median_6d']:.6g}, "
        f"coverage95={100*n['coverage_6d_95']:.2f}%, "
        f"coverage99={100*n['coverage_6d_99']:.2f}%"
    )
    print(
        "NEES trans/rot mean: "
        f"{n['mean_translation_3d']:.6g} / "
        f"{n['mean_rotation_3d']:.6g} "
        "(Gaussian reference expectation 3 / 3)"
    )
    c = result["error_uncertainty_correlation"]
    print(
        "sigma-vs-error Spearman: "
        f"translation={c['translation_spearman']}, "
        f"rotation={c['rotation_spearman']}"
    )
    print(
        "whitened error mean: "
        + " ".join(f"{v:.4g}" for v in result["whitened_error"]["mean"])
    )
    print(
        "whitened error std:  "
        + " ".join(f"{v:.4g}" for v in result["whitened_error"]["std"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="One or more M2-A diagnostic directories.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    specs = parse_specs(args.sequence)
    report = {
        "model": "m2a_motion_prior_same_object_consistency",
        "pose_convention": "world_to_camera_right_perturbation",
        "error_definition": "Log(inv(T_motion_prior) @ T_gt)",
        "covariance": "P_abs_prior_right",
        "chi_square_note": (
            "Coverage thresholds are descriptive references only; trajectory "
            "frames are temporally correlated and are not iid calibration draws."
        ),
        "sequences": {},
    }

    pooled = []
    for name, directory in specs:
        rows, skipped = load_sequence(name, directory)
        result = summarize(rows, "P")
        raw_result = summarize(rows, "P_raw_chain")
        report["sequences"][name] = {
            "directory": str(directory),
            "skipped": skipped,
            "selected_prior": result,
            "raw_recursive_chain": raw_result,
        }
        pooled.extend(rows)
        print_summary(name, result, skipped)

    pooled_result = summarize(pooled, "P")
    report["pooled_selected_prior"] = pooled_result

    if len(specs) > 1:
        print_summary(
            "POOLED (descriptive only)",
            pooled_result,
            {"missing_target": 0, "nonfinite": 0},
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
