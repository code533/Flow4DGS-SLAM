#!/usr/bin/env python3
"""Summarize M2-A2 finite-difference sensitivity diagnostics.

Example:
    python scripts/analyze_m2a2_fd_sensitivity.py \
        results/m2a_pose_uncertainty
"""

import argparse
from pathlib import Path

import torch


def fmt_vec(x):
    return "[" + ", ".join(f"{float(v):.4g}" for v in x) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    files = sorted(args.directory.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {args.directory}")

    found = 0
    baseline_j = []
    baseline_cond = []

    for path in files:
        d = torch.load(path, map_location="cpu")

        jrms = d.get("m2a2_j_rms", None)
        bcond = d.get("m2a2_bread_condition", None)
        if jrms is not None:
            baseline_j.append(jrms.double())
        if bcond is not None:
            baseline_cond.append(float(bcond))

        sweep = d.get("m2a2_fd_audit", None)
        if not sweep:
            continue

        found += 1
        print("\n" + "=" * 76)
        print(f"frame {int(d['frame'])}")
        print("=" * 76)

        for family in ("translation", "rotation"):
            rows = sweep.get(family, [])
            print(f"\n{family} epsilon sweep")
            print(
                "eps        diff_rms[6]                                  "
                "J_rms[6]                                     bread_cond"
            )
            for row in rows:
                eig = row["bread_eigenvalues"].double()
                print(
                    f"{row['eps']:<10.3g} "
                    f"{fmt_vec(row['diff_rms'])!s:<46} "
                    f"{fmt_vec(row['j_rms'])!s:<46} "
                    f"{float(row['bread_condition']):.4g}"
                )
                print(
                    "           bread eig: "
                    + fmt_vec(eig)
                )

    print("\n" + "=" * 76)
    print("M2-A2 FD sensitivity summary")
    print("=" * 76)
    print(f"audited frames: {found}")

    if baseline_j:
        J = torch.stack(baseline_j)
        print(
            "baseline J_rms median by DoF: "
            + fmt_vec(J.median(dim=0).values)
        )
        print(
            "baseline J_rms mean by DoF:   "
            + fmt_vec(J.mean(dim=0))
        )

    if baseline_cond:
        c = torch.tensor(baseline_cond, dtype=torch.float64)
        print(
            "baseline bread condition: "
            f"median={float(c.median()):.4g}, "
            f"mean={float(c.mean()):.4g}, "
            f"max={float(c.max()):.4g}"
        )

    if found == 0:
        print(
            "No epsilon-sweep payloads found. Enable "
            "Uncertainty.m2a2_fd_audit and rerun selected frames."
        )


if __name__ == "__main__":
    main()
