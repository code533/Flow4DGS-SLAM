#!/usr/bin/env python3
"""Audit M2-A2 covariance after removing the trajectory SE(3) gauge.

Flow4DGS evaluates ATE after a rigid SE(3) trajectory alignment.  The saved
M2-A2 poses, however, live in the estimator's world gauge.  Directly computing
Log(T_est^{-1} T_gt) therefore mixes arbitrary global gauge offset with local
pose error.

This script fits one no-scale Kabsch/Umeyama alignment A per sequence in the
camera-to-world convention:
    C_est_aligned = A C_est
and converts it back to the saved world-to-camera convention:
    T_est_aligned = T_est A^{-1}.

For the saved right perturbation
    T = Tbar Exp(delta),
the deterministic gauge change T' = T A^{-1} maps
    delta' = Ad_A delta,
so the covariance is transformed consistently:
    P' = Ad_A P Ad_A^T.

The resulting NEES is an alignment-conditioned diagnostic: because A is
estimated from the same GT trajectory, do not interpret chi-square coverage as
an exact independent-sample statistical test.  Its purpose here is to verify
whether previous raw GT errors were dominated by gauge mismatch.

Diagnostic only; no runtime behavior is changed.
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

from utils.m2_uncertainty import SE3_log, adjoint_SE3


CHI2_6_95 = 12.591587


def parse_specs(values):
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=M2_DIR, got {value!r}")
        name, path = value.split("=", 1)
        out.append((name.strip(), Path(path).expanduser()))
    return out


def sym(P):
    return 0.5 * (P + P.T)


def stable_nees(e, P):
    P = sym(P)
    try:
        return float(e @ torch.linalg.solve(P, e))
    except RuntimeError:
        return float(e @ (torch.linalg.pinv(P) @ e))


def quantile(x, q):
    return float(np.quantile(np.asarray(x, dtype=np.float64), q))


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
        "rmse": float(np.sqrt(np.mean(x * x))),
    }


def fit_se3_alignment(est_xyz, gt_xyz):
    """Return A mapping estimated c2w world coordinates into GT coordinates."""
    xbar = est_xyz.mean(axis=0)
    ybar = gt_xyz.mean(axis=0)
    X = est_xyz - xbar
    Y = gt_xyz - ybar
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = ybar - R @ xbar
    A = np.eye(4, dtype=np.float64)
    A[:3, :3] = R
    A[:3, 3] = t
    return A


def load_sequence(name, directory):
    samples = []
    for path in sorted(directory.glob("*.pt")):
        d = torch.load(path, map_location="cpu")
        T = d.get("T_final")
        G = d.get("T_gt")
        P = d.get("P_track_right_raw")
        source = "P_track_right_raw"
        if P is None:
            P = d.get("P_track_right")
            source = "P_track_right"
        if not all(isinstance(x, torch.Tensor) for x in (T, G, P)):
            continue
        T = T.double()
        G = G.double()
        P = sym(P.double())
        if (
            tuple(T.shape) != (4, 4)
            or tuple(G.shape) != (4, 4)
            or tuple(P.shape) != (6, 6)
        ):
            continue
        if not bool(
            torch.isfinite(T).all()
            and torch.isfinite(G).all()
            and torch.isfinite(P).all()
        ):
            continue
        samples.append({
            "frame": int(d.get("frame", int(path.stem))),
            "T": T,
            "G": G,
            "P": P,
            "source": source,
        })
    if len(samples) < 3:
        raise RuntimeError(f"Need >=3 usable samples in {directory}")
    return samples


def audit(name, directory):
    samples = load_sequence(name, directory)

    C_est = np.stack([
        torch.linalg.inv(s["T"])[:3, 3].numpy() for s in samples
    ])
    C_gt = np.stack([
        torch.linalg.inv(s["G"])[:3, 3].numpy() for s in samples
    ])
    A_np = fit_se3_alignment(C_est, C_gt)
    A = torch.from_numpy(A_np).double()
    Ainv = torch.linalg.inv(A)
    AdA = adjoint_SE3(A)

    raw_et, raw_er, raw_nees = [], [], []
    ali_et, ali_er, ali_nees = [], [], []
    center_err = []

    for s in samples:
        e0 = SE3_log(torch.linalg.inv(s["T"]) @ s["G"])
        raw_et.append(float(torch.linalg.norm(e0[:3])))
        raw_er.append(float(torch.linalg.norm(e0[3:])))
        raw_nees.append(stable_nees(e0, s["P"]))

        T_aligned = s["T"] @ Ainv
        P_aligned = sym(AdA @ s["P"] @ AdA.T)
        e = SE3_log(torch.linalg.inv(T_aligned) @ s["G"])
        ali_et.append(float(torch.linalg.norm(e[:3])))
        ali_er.append(float(torch.linalg.norm(e[3:])))
        ali_nees.append(stable_nees(e, P_aligned))

        C = torch.linalg.inv(T_aligned)[:3, 3]
        Cg = torch.linalg.inv(s["G"])[:3, 3]
        center_err.append(float(torch.linalg.norm(C - Cg)))

    out = {
        "name": name,
        "directory": str(directory),
        "count": len(samples),
        "covariance_source": sorted(set(s["source"] for s in samples)),
        "alignment_A_c2w_est_to_gt": A_np.tolist(),
        "raw_log_translation_error_m": stats(raw_et),
        "raw_log_rotation_error_rad": stats(raw_er),
        "aligned_log_translation_error_m": stats(ali_et),
        "aligned_log_rotation_error_rad": stats(ali_er),
        "aligned_camera_center_error_m": stats(center_err),
        "raw_nees_6d": stats(raw_nees),
        "aligned_nees_6d": stats(ali_nees),
        "raw_coverage_chi2_6_95": float(
            np.mean(np.asarray(raw_nees) <= CHI2_6_95)
        ),
        "aligned_coverage_chi2_6_95": float(
            np.mean(np.asarray(ali_nees) <= CHI2_6_95)
        ),
        "note": (
            "Aligned NEES is gauge-conditioned because the SE(3) alignment "
            "is estimated from this sequence's GT trajectory."
        ),
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sequence", nargs="+", required=True, metavar="NAME=M2_DIR"
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("results/m2a2_gauge_alignment_audit.json"),
    )
    args = p.parse_args()

    report = {}
    print("=" * 84)
    print("M2-A2 gauge-aligned covariance audit")
    print("=" * 84)
    print(
        "SE(3) alignment removes one global trajectory gauge per sequence; "
        "no scale correction is used."
    )

    for name, directory in parse_specs(args.sequence):
        r = audit(name, directory)
        report[name] = r
        print(f"\n{name}: {r['count']} frames")
        print(
            "  raw Log error median [t/r]: "
            f"{r['raw_log_translation_error_m']['median']:.6g} m / "
            f"{r['raw_log_rotation_error_rad']['median']:.6g} rad"
        )
        print(
            "  aligned Log error median [t/r]: "
            f"{r['aligned_log_translation_error_m']['median']:.6g} m / "
            f"{r['aligned_log_rotation_error_rad']['median']:.6g} rad"
        )
        print(
            "  aligned center error median/p95: "
            f"{r['aligned_camera_center_error_m']['median']:.6g} / "
            f"{r['aligned_camera_center_error_m']['p95']:.6g} m"
        )
        print(
            "  NEES mean raw -> aligned: "
            f"{r['raw_nees_6d']['mean']:.6g} -> "
            f"{r['aligned_nees_6d']['mean']:.6g}"
        )
        print(
            "  chi2(6) 95% reference coverage raw -> aligned: "
            f"{100*r['raw_coverage_chi2_6_95']:.2f}% -> "
            f"{100*r['aligned_coverage_chi2_6_95']:.2f}%"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
