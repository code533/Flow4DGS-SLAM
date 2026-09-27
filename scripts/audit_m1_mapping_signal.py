#!/usr/bin/env python3
"""Audit an M1-derived scalar before uncertainty-aware Gaussian mapping.

This script is deliberately offline: it does not modify tracking, mapping, pose
updates, keyframe selection, or Gaussian insertion. It tests whether a scalar
constructed from calibrated M1 relative-motion covariance retains useful
held-out ranking information before that scalar is allowed to affect SLAM.

For a held-out LOSO fold, calibration is fitted elsewhere on the fold's
training sequences. The same frozen calibration is applied to training and
held-out raw M1 covariances. Training-only median marginal sigmas define the
dimensionless scalar

    u^2 = 0.5 * [(sigma_t / m_t)^2 + (sigma_r / m_r)^2].

Candidate confidence is c = 1 / (u^2 + eps). The script reports both
sequence-global normalized weights and a rolling-window proxy. The latter is
NOT the exact backend mapping-window distribution because current M1
diagnostics do not record keyframe-window membership.

Example:
    python scripts/audit_m1_mapping_signal.py \
        --sequence placing_box=/path/box1/m1_pose_uncertainty \
                   placing_box2=/path/box2/m1_pose_uncertainty \
                   placing_box3=/path/box3/m1_pose_uncertainty \
        --calibration results/m1_loso_bonn_box123.json \
        --block-size 32 \
        --mode diag \
        --window-size 8 \
        --clip-min 0.25 \
        --clip-max 4.0 \
        --output results/m1_mapping_signal_bonn_loso.json
"""

import argparse
import json
import math
from pathlib import Path

import torch

from calibrate_m1_multiseq import (
    load_sequence,
    parse_sequence_specs,
    pearson,
    symmetric_sqrt_and_inv,
    symmetrize,
)


def finite(values):
    return [float(v) for v in values if math.isfinite(float(v))]


def stats(values):
    x = torch.tensor(finite(values), dtype=torch.float64)
    if x.numel() == 0:
        return {"count": 0}
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=x.dtype)
    qq = torch.quantile(x, q)
    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "q05": float(qq[0]),
        "q25": float(qq[1]),
        "median": float(qq[2]),
        "q75": float(qq[3]),
        "q95": float(qq[4]),
        "max": float(x.max()),
    }


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
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return float("nan")
    xx, yy = zip(*pairs)
    return pearson(rankdata(xx), rankdata(yy))


def finite_mean(values):
    x = finite(values)
    if not x:
        return float("nan")
    return float(torch.tensor(x, dtype=torch.float64).mean())


def calibration_matrix(report, held_out, mode):
    if report.get("mode") != "loso":
        raise ValueError(
            "This audit currently requires a LOSO calibration report so each "
            "held-out sequence uses training-only calibration."
        )
    folds = report.get("folds", {})
    if held_out not in folds:
        raise KeyError(f"Calibration report has no LOSO fold: {held_out}")

    fold = folds[held_out]
    variants = (
        fold.get("calibration", {})
        .get("whitened", {})
        .get("variants", {})
    )
    if mode not in variants:
        raise KeyError(
            f"Fold {held_out} has no whitened calibration variant '{mode}'"
        )
    C = torch.tensor(variants[mode]["C"], dtype=torch.float64)
    if C.shape != (6, 6):
        raise ValueError(
            f"Expected 6x6 calibration matrix for {held_out}, got {tuple(C.shape)}"
        )
    return fold, symmetrize(C)


def calibrated_covariance(P, C, eig_floor_rel):
    root, _ = symmetric_sqrt_and_inv(
        P.double(), eig_floor_rel=eig_floor_rel
    )
    return symmetrize(root @ C @ root)


def marginal_sigmas(sample, C, eig_floor_rel):
    P = calibrated_covariance(sample["P"], C, eig_floor_rel)
    diag = torch.diagonal(P).clamp_min(0.0)
    sigma_t = float(torch.sqrt(diag[:3].mean()))
    sigma_r = float(torch.sqrt(diag[3:].mean()))
    return sigma_t, sigma_r


def scalar_u(sigma_t, sigma_r, ref_t, ref_r):
    return math.sqrt(
        0.5
        * (
            (float(sigma_t) / float(ref_t)) ** 2
            + (float(sigma_r) / float(ref_r)) ** 2
        )
    )


def tertile_summary(rows):
    if len(rows) < 3:
        return {}
    ordered = sorted(rows, key=lambda r: r["u"])
    n = len(ordered)
    cuts = [0, n // 3, (2 * n) // 3, n]
    names = ["low", "middle", "high"]
    out = {}
    for name, lo, hi in zip(names, cuts[:-1], cuts[1:]):
        group = ordered[lo:hi]
        out[name] = {
            "count": len(group),
            "u": stats([r["u"] for r in group]),
            "error_t": stats([r["error_t"] for r in group]),
            "error_r": stats([r["error_r"] for r in group]),
        }
    out["median_error_t_monotonic"] = (
        out["low"]["error_t"]["median"]
        <= out["middle"]["error_t"]["median"]
        <= out["high"]["error_t"]["median"]
    )
    out["median_error_r_monotonic"] = (
        out["low"]["error_r"]["median"]
        <= out["middle"]["error_r"]["median"]
        <= out["high"]["error_r"]["median"]
    )
    return out


def normalize_weights(confidences, clip_min, clip_max):
    c = torch.tensor(confidences, dtype=torch.float64)
    if c.numel() == 0:
        return []
    mean = c.mean().clamp_min(1e-18)
    w = c / mean
    if clip_min is not None or clip_max is not None:
        lo = -float("inf") if clip_min is None else float(clip_min)
        hi = float("inf") if clip_max is None else float(clip_max)
        w = torch.clamp(w, min=lo, max=hi)
        # Re-normalize after clipping so the candidate intervention preserves
        # average loss scale inside the window.
        w = w / w.mean().clamp_min(1e-18)
    return w.tolist()


def rolling_window_weight_proxy(rows, window_size, clip_min, clip_max):
    """Return proxy weights over contiguous M1 frames.

    Backend mapping windows contain keyframes, whereas M1 diagnostics are saved
    per frame and do not encode exact keyframe-window membership. This proxy
    therefore must not be reported as the runtime mapping-weight distribution.
    """
    rows = sorted(rows, key=lambda r: r["frame"])
    raw_all = []
    clipped_all = []
    raw_clip_hits = 0
    raw_count = 0

    for end in range(len(rows)):
        start = max(0, end - window_size + 1)
        c = [r["confidence"] for r in rows[start : end + 1]]
        raw = normalize_weights(c, None, None)
        clipped = normalize_weights(c, clip_min, clip_max)
        raw_all.extend(raw)
        clipped_all.extend(clipped)

        if clip_min is not None or clip_max is not None:
            for w in raw:
                hit = False
                if clip_min is not None and w < clip_min:
                    hit = True
                if clip_max is not None and w > clip_max:
                    hit = True
                raw_clip_hits += int(hit)
                raw_count += 1

    return {
        "note": (
            "Contiguous-frame proxy only; M1 diagnostics do not record exact "
            "backend keyframe-window membership."
        ),
        "window_size": int(window_size),
        "raw_normalized": stats(raw_all),
        "post_clip_renormalized": stats(clipped_all),
        "candidate_clip_fraction_before_clipping": (
            float(raw_clip_hits / raw_count) if raw_count else float("nan")
        ),
    }


def audit_fold(
    held_out,
    all_samples,
    report,
    mode,
    eig_floor_rel,
    eps,
    window_size,
    clip_min,
    clip_max,
):
    fold, C = calibration_matrix(report, held_out, mode)
    train_names = list(fold.get("train_sequences", []))
    if held_out in train_names:
        raise ValueError(
            f"Invalid LOSO report: held-out sequence {held_out} is in training set"
        )
    missing = [name for name in train_names if name not in all_samples]
    if missing:
        raise KeyError(
            f"Fold {held_out} needs training sequences not supplied via "
            f"--sequence: {missing}"
        )

    train_sigmas_t = []
    train_sigmas_r = []
    for name in train_names:
        for sample in all_samples[name]:
            st, sr = marginal_sigmas(sample, C, eig_floor_rel)
            if math.isfinite(st) and math.isfinite(sr) and st > 0 and sr > 0:
                train_sigmas_t.append(st)
                train_sigmas_r.append(sr)

    if not train_sigmas_t:
        raise RuntimeError(f"No finite training sigmas for fold {held_out}")

    ref_t = float(torch.tensor(train_sigmas_t, dtype=torch.float64).median())
    ref_r = float(torch.tensor(train_sigmas_r, dtype=torch.float64).median())
    if ref_t <= 0 or ref_r <= 0:
        raise RuntimeError(
            f"Non-positive training reference scale for fold {held_out}: "
            f"m_t={ref_t}, m_r={ref_r}"
        )

    rows = []
    for sample in all_samples[held_out]:
        st, sr = marginal_sigmas(sample, C, eig_floor_rel)
        u = scalar_u(st, sr, ref_t, ref_r)
        confidence = 1.0 / (u * u + eps)
        err = sample["error"]
        error_t = float(torch.linalg.norm(err[:3]))
        error_r = float(torch.linalg.norm(err[3:]))
        if not all(
            math.isfinite(v)
            for v in (st, sr, u, confidence, error_t, error_r)
        ):
            continue
        rows.append(
            {
                "frame": int(sample["frame"]),
                "sigma_t": st,
                "sigma_r": sr,
                "u": u,
                "confidence": confidence,
                "error_t": error_t,
                "error_r": error_r,
            }
        )

    if not rows:
        raise RuntimeError(f"No finite held-out samples for fold {held_out}")

    u = [r["u"] for r in rows]
    et = [r["error_t"] for r in rows]
    er = [r["error_r"] for r in rows]
    conf = [r["confidence"] for r in rows]

    raw_global = normalize_weights(conf, None, None)
    clipped_global = normalize_weights(conf, clip_min, clip_max)
    clip_hits = 0
    if clip_min is not None or clip_max is not None:
        for w in raw_global:
            hit = (
                (clip_min is not None and w < clip_min)
                or (clip_max is not None and w > clip_max)
            )
            clip_hits += int(hit)

    tertiles = tertile_summary(rows)
    rho_t = spearman(u, et)
    rho_r = spearman(u, er)

    return {
        "held_out": held_out,
        "train_sequences": train_names,
        "count": len(rows),
        "calibration_mode": mode,
        "reference_scales_training_only": {
            "median_sigma_t_m": ref_t,
            "median_sigma_r_rad": ref_r,
            "training_count": len(train_sigmas_t),
        },
        "signal": {
            "u": stats(u),
            "sigma_t_m": stats([r["sigma_t"] for r in rows]),
            "sigma_r_rad": stats([r["sigma_r"] for r in rows]),
            "spearman_u_error_t": rho_t,
            "spearman_u_error_r": rho_r,
            "positive_association_t": bool(math.isfinite(rho_t) and rho_t > 0),
            "positive_association_r": bool(math.isfinite(rho_r) and rho_r > 0),
        },
        "tertiles": tertiles,
        "candidate_weights": {
            "formula": "c=1/(u^2+eps); w=c/mean(c)",
            "eps": eps,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "sequence_global_proxy": {
                "note": (
                    "Diagnostic proxy only; runtime method will normalize "
                    "within the actual mapping window."
                ),
                "raw_normalized": stats(raw_global),
                "post_clip_renormalized": stats(clipped_global),
                "candidate_clip_fraction_before_clipping": (
                    float(clip_hits / len(raw_global))
                    if raw_global
                    else float("nan")
                ),
            },
            "rolling_window_proxy": rolling_window_weight_proxy(
                rows, window_size, clip_min, clip_max
            ),
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline audit of the M1 mapping-reliability scalar."
    )
    parser.add_argument(
        "--sequence",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="M1 diagnostic directories. Supply all sequences used by LOSO.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="LOSO JSON from scripts/calibrate_m1_multiseq.py.",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=["diag", "block", "full"],
        default="diag",
        help="Frozen whitened covariance calibration variant.",
    )
    parser.add_argument("--eig-floor-rel", type=float, default=1e-10)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Rolling contiguous-frame proxy size; Bonn/TUM backend default is 8.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.25,
        help="Candidate normalized-weight lower clip for the proxy audit.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=4.0,
        help="Candidate normalized-weight upper clip for the proxy audit.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1")
    if args.eps <= 0:
        raise ValueError("--eps must be > 0")
    if args.clip_min is not None and args.clip_min <= 0:
        raise ValueError("--clip-min must be > 0")
    if (
        args.clip_min is not None
        and args.clip_max is not None
        and args.clip_min > args.clip_max
    ):
        raise ValueError("--clip-min must be <= --clip-max")

    specs = parse_sequence_specs(args.sequence)
    names = [name for name, _ in specs]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate sequence names in --sequence")

    with args.calibration.open("r") as f:
        report = json.load(f)

    all_samples = {}
    covariance_sources = {}
    for name, directory in specs:
        samples, source = load_sequence(name, directory, args.block_size)
        all_samples[name] = samples
        covariance_sources[name] = source

    folds = {}
    for held_out in names:
        folds[held_out] = audit_fold(
            held_out=held_out,
            all_samples=all_samples,
            report=report,
            mode=args.mode,
            eig_floor_rel=args.eig_floor_rel,
            eps=args.eps,
            window_size=args.window_size,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )

    rho_t = [folds[n]["signal"]["spearman_u_error_t"] for n in names]
    rho_r = [folds[n]["signal"]["spearman_u_error_r"] for n in names]
    monotonic_t = [
        bool(folds[n].get("tertiles", {}).get("median_error_t_monotonic", False))
        for n in names
    ]
    monotonic_r = [
        bool(folds[n].get("tertiles", {}).get("median_error_r_monotonic", False))
        for n in names
    ]

    output = {
        "method": "m1_dimensionless_mapping_signal_audit",
        "block_size": args.block_size,
        "calibration_file": str(args.calibration),
        "calibration_mode": args.mode,
        "eig_floor_rel": args.eig_floor_rel,
        "sequence_paths": {name: str(path) for name, path in specs},
        "covariance_sources": covariance_sources,
        "definition": {
            "sigma_t": "sqrt(trace(P_cal[0:3,0:3])/3)",
            "sigma_r": "sqrt(trace(P_cal[3:6,3:6])/3)",
            "m_t": "training-only median calibrated sigma_t",
            "m_r": "training-only median calibrated sigma_r",
            "u2": "0.5*((sigma_t/m_t)^2+(sigma_r/m_r)^2)",
            "confidence": "1/(u^2+eps)",
            "weight": "confidence/mean_window(confidence)",
        },
        "folds": folds,
        "aggregate": {
            "spearman_u_error_t_macro_mean": finite_mean(rho_t),
            "spearman_u_error_r_macro_mean": finite_mean(rho_r),
            "positive_association_t_all_folds": all(
                math.isfinite(v) and v > 0 for v in rho_t
            ),
            "positive_association_r_all_folds": all(
                math.isfinite(v) and v > 0 for v in rho_r
            ),
            "tertile_median_error_t_monotonic_all_folds": all(monotonic_t),
            "tertile_median_error_r_monotonic_all_folds": all(monotonic_r),
        },
        "interpretation_guardrail": (
            "This audit evaluates a local relative-motion uncertainty signal. "
            "It does not establish absolute-pose reliability or end-task SLAM "
            "improvement. Rolling-window weights are a contiguous-frame proxy, "
            "not exact backend keyframe-window weights."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2, allow_nan=True)

    print("M1 mapping-signal audit")
    print("  calibration:", args.calibration)
    print("  mode:", args.mode)
    print("  block size:", args.block_size)
    for name in names:
        fold = folds[name]
        sig = fold["signal"]
        tert = fold["tertiles"]
        proxy = fold["candidate_weights"]["rolling_window_proxy"]
        print(f"\n[{name}] n={fold['count']}")
        print(
            "  train refs: "
            f"m_t={fold['reference_scales_training_only']['median_sigma_t_m']:.6g} m, "
            f"m_r={fold['reference_scales_training_only']['median_sigma_r_rad']:.6g} rad"
        )
        print(
            "  Spearman(u,error): "
            f"t={sig['spearman_u_error_t']:.4f}, "
            f"r={sig['spearman_u_error_r']:.4f}"
        )
        if tert:
            print(
                "  tertile median error_t: "
                f"{tert['low']['error_t']['median']:.6g} -> "
                f"{tert['middle']['error_t']['median']:.6g} -> "
                f"{tert['high']['error_t']['median']:.6g}"
            )
            print(
                "  tertile median error_r: "
                f"{tert['low']['error_r']['median']:.6g} -> "
                f"{tert['middle']['error_r']['median']:.6g} -> "
                f"{tert['high']['error_r']['median']:.6g}"
            )
        print(
            "  rolling weight proxy raw q05/median/q95: "
            f"{proxy['raw_normalized']['q05']:.4f} / "
            f"{proxy['raw_normalized']['median']:.4f} / "
            f"{proxy['raw_normalized']['q95']:.4f}"
        )
        print(
            "  candidate clip-hit fraction: "
            f"{proxy['candidate_clip_fraction_before_clipping']:.3%}"
        )

    agg = output["aggregate"]
    print("\n[aggregate]")
    print(
        "  macro Spearman(u,error): "
        f"t={agg['spearman_u_error_t_macro_mean']:.4f}, "
        f"r={agg['spearman_u_error_r_macro_mean']:.4f}"
    )
    print(
        "  positive association in every fold: "
        f"t={agg['positive_association_t_all_folds']}, "
        f"r={agg['positive_association_r_all_folds']}"
    )
    print(
        "  tertile median error monotonic in every fold: "
        f"t={agg['tertile_median_error_t_monotonic_all_folds']}, "
        f"r={agg['tertile_median_error_r_monotonic_all_folds']}"
    )
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
