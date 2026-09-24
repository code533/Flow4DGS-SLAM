#!/usr/bin/env python3
"""Summarize M1 relative-pose covariance diagnostics.

Usage:
    python scripts/analyze_m1_pose_uncertainty.py /path/to/run/m1_pose_uncertainty

This script intentionally does not require ground-truth poses. It performs
basic numerical checks and reports per-frame uncertainty/conditioning. NEES
analysis will be added once the relative-pose GT convention is wired in.
"""

import argparse
import math
from pathlib import Path

import torch


def _scalar(x):
    if torch.is_tensor(x):
        return float(x.item())
    return float(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--psd-tol", type=float, default=1e-9)
    args = parser.parse_args()

    files = sorted(args.directory.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt diagnostics found in {args.directory}")

    rows = []
    bad_finite = 0
    bad_symmetry = 0
    bad_psd = 0

    for path in files:
        data = torch.load(path, map_location="cpu")
        P = data["P_xi_raw"].double()

        finite = bool(torch.isfinite(P).all())
        symmetric = bool(torch.allclose(P, P.T, atol=1e-8, rtol=1e-6))
        eig = torch.linalg.eigvalsh(0.5 * (P + P.T))
        psd = bool(eig.min() >= -args.psd_tol)

        bad_finite += int(not finite)
        bad_symmetry += int(not symmetric)
        bad_psd += int(not psd)

        sig_t = data["sigma_trans_m"].double()
        sig_r = data["sigma_rot_rad"].double()

        rows.append(
            {
                "frame": int(data["frame"]),
                "trans_rms": float(torch.sqrt((sig_t * sig_t).mean())),
                "rot_rms": float(torch.sqrt((sig_r * sig_r).mean())),
                "condition": _scalar(data["condition"]),
                "kappa": _scalar(data["kappa"]),
                "fb": _scalar(data["fb_error_median_px"]),
                "maha": _scalar(data["maha_median"]),
                "n": int(data["num_pixels"]),
                "eig_min": float(eig.min()),
                "eig_max": float(eig.max()),
            }
        )

    print(f"frames: {len(rows)}")
    print(
        "numerical checks: "
        f"nonfinite={bad_finite}, nonsymmetric={bad_symmetry}, nonPSD={bad_psd}"
    )

    finite_rows = [
        r for r in rows
        if math.isfinite(r["condition"]) and math.isfinite(r["trans_rms"])
    ]
    if finite_rows:
        trans = torch.tensor([r["trans_rms"] for r in finite_rows])
        rot = torch.tensor([r["rot_rms"] for r in finite_rows])
        cond = torch.tensor([r["condition"] for r in finite_rows])
        kappa = torch.tensor([r["kappa"] for r in finite_rows])
        fb = torch.tensor([r["fb"] for r in finite_rows])

        def stats(name, x):
            print(
                f"{name:>14s}: "
                f"median={x.median().item():.6g}, "
                f"mean={x.mean().item():.6g}, "
                f"max={x.max().item():.6g}"
            )

        stats("sigma_t_rms", trans)
        stats("sigma_r_rms", rot)
        stats("condition", cond)
        stats("kappa", kappa)
        stats("fb_error_px", fb)

    print("\nTop frames by translation uncertainty:")
    for r in sorted(rows, key=lambda x: x["trans_rms"], reverse=True)[:10]:
        print(
            f"frame={r['frame']:6d} "
            f"sigma_t={r['trans_rms']:.4e} "
            f"sigma_r={r['rot_rms']:.4e} "
            f"cond={r['condition']:.3e} "
            f"kappa={r['kappa']:.3e} "
            f"fb={r['fb']:.3f} "
            f"N={r['n']}"
        )


if __name__ == "__main__":
    main()
