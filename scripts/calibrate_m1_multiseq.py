#!/usr/bin/env python3
"""Cross-sequence calibration for M1 pose covariance.

Example:

    python scripts/calibrate_m1_multiseq.py \
        --block-size 32 \
        --train walking_xyz=/path/run1/m1_pose_uncertainty \
                sitting_static=/path/run2/m1_pose_uncertainty \
        --test bonn_placing=/path/run3/m1_pose_uncertainty \
               sitting_rpy=/path/run4/m1_pose_uncertainty \
        --output results/m1_calibration_block32.json

The calibration model is

    P_cal = S P S^T,
    S = diag(s_t, s_t, s_t, s_r, s_r, s_r).

s_t and s_r are fitted only on the training sequences by matching the mean
translation- and rotation-marginal NEES to their expected 3-DoF value.

This script intentionally keeps the calibration model simple. It is meant to
answer whether a sequence-independent translation/rotation scale can make the
correlation-aware M1 covariance transferable before introducing any additional
model-discrepancy term Q_model.
"""

import argparse
import json
import math
from pathlib import Path

import torch


CHI2_3_95 = 7.814728
CHI2_3_99 = 11.344867
CHI2_6_95 = 12.591587
CHI2_6_99 = 16.811894


def skew(w):
    W = torch.zeros((3, 3), dtype=w.dtype)
    W[0, 1], W[0, 2] = -w[2], w[1]
    W[1, 0], W[1, 2] = w[2], -w[0]
    W[2, 0], W[2, 1] = -w[1], w[0]
    return W


def so3_log(R):
    c = torch.clamp((torch.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    a = torch.acos(c)
    vee = torch.tensor(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=R.dtype,
    )
    if a < 1e-8:
        return 0.5 * vee
    return a / (2.0 * torch.sin(a)) * vee


def left_jacobian_so3(w):
    a = w.norm()
    W = skew(w)
    I = torch.eye(3, dtype=w.dtype)
    if a < 1e-8:
        return I + 0.5 * W + (1.0 / 6.0) * (W @ W)
    a2 = a * a
    return (
        I
        + ((1.0 - torch.cos(a)) / a2) * W
        + ((a - torch.sin(a)) / (a2 * a)) * (W @ W)
    )


def se3_log(T):
    w = so3_log(T[:3, :3])
    V = left_jacobian_so3(w)
    t = T[:3, 3]
    try:
        rho = torch.linalg.solve(V, t)
    except RuntimeError:
        rho = torch.linalg.pinv(V) @ t
    return torch.cat([rho, w])


def parse_sequence_specs(values):
    specs = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Expected NAME=PATH sequence spec, got: {value}"
            )
        name, path = value.split("=", 1)
        name = name.strip()
        path = Path(path).expanduser()
        if not name:
            raise ValueError(f"Empty sequence name in spec: {value}")
        specs.append((name, path))
    return specs


def select_covariance(data, block_size):
    if block_size is None:
        if "P_xi_cluster" in data:
            return data["P_xi_cluster"].double(), "P_xi_cluster"
        return data["P_xi_raw"].double(), "P_xi_raw"

    multiscale = data.get("P_xi_clusters", {})
    for key in (block_size, str(block_size)):
        if key in multiscale:
            return multiscale[key].double(), f"P_xi_clusters[{block_size}]"

    saved_block = int(data.get("cluster_block_size", -1))
    if saved_block == block_size and "P_xi_cluster" in data:
        return data["P_xi_cluster"].double(), "P_xi_cluster"

    raise KeyError(
        f"Requested block size {block_size}, but diagnostic file does not "
        "contain that covariance. Re-run M1 with cluster_block_sizes including "
        f"{block_size}."
    )


def stable_nees(err, P):
    P = 0.5 * (P + P.T)
    try:
        return float(err @ torch.linalg.solve(P, err))
    except RuntimeError:
        return float(err @ (torch.linalg.pinv(P) @ err))


def load_sequence(name, directory, block_size):
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No M1 .pt files in {directory}")

    samples = []
    source_key = None

    for path in files:
        data = torch.load(path, map_location="cpu")
        if "T_rel_gt" not in data:
            continue

        P, key = select_covariance(data, block_size)
        source_key = source_key or key

        xi_est = data.get("xi_baseline", data["xi_raw"]).double()
        xi_gt = se3_log(data["T_rel_gt"].double())
        err = xi_est - xi_gt

        P = 0.5 * (P + P.T)
        if not bool(torch.isfinite(P).all()):
            continue

        samples.append(
            {
                "sequence": name,
                "frame": int(data["frame"]),
                "error": err,
                "P": P,
                "fb": float(data["fb_error_median_px"]),
                "maha": float(data["maha_median"]),
                "condition": float(data["condition"]),
                "num_pixels": int(data["num_pixels"]),
            }
        )

    if not samples:
        raise RuntimeError(
            f"No GT-capable finite M1 diagnostics found in {directory}"
        )

    return samples, source_key


def marginal_nees(sample, sl):
    err = sample["error"][sl]
    P = sample["P"][sl, sl]
    return stable_nees(err, P)


def fit_scales(samples):
    if not samples:
        raise ValueError("No training samples")

    nees_t = torch.tensor(
        [marginal_nees(s, slice(0, 3)) for s in samples],
        dtype=torch.float64,
    )
    nees_r = torch.tensor(
        [marginal_nees(s, slice(3, 6)) for s in samples],
        dtype=torch.float64,
    )

    # For P' = s^2 P, marginal NEES scales as 1/s^2.
    alpha_t = max(float(nees_t.mean() / 3.0), 1e-12)
    alpha_r = max(float(nees_r.mean() / 3.0), 1e-12)
    s_t = math.sqrt(alpha_t)
    s_r = math.sqrt(alpha_r)

    return {
        "s_t": s_t,
        "s_r": s_r,
        "alpha_t": alpha_t,
        "alpha_r": alpha_r,
        "train_mean_nees_t_raw": float(nees_t.mean()),
        "train_mean_nees_r_raw": float(nees_r.mean()),
    }


def scale_covariance(P, s_t, s_r):
    S = torch.diag(
        torch.tensor(
            [s_t, s_t, s_t, s_r, s_r, s_r],
            dtype=P.dtype,
        )
    )
    return S @ P @ S


def covariance_sigma(P, sl):
    d = torch.diagonal(P)[sl].clamp_min(0.0)
    return float(torch.sqrt(d.mean()))


def pearson(x, y):
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum())
    if den <= 0:
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


def summarize(samples, s_t=1.0, s_r=1.0):
    nees6 = []
    nees_t = []
    nees_r = []
    sig_t = []
    sig_r = []
    err_t = []
    err_r = []

    for sample in samples:
        P = scale_covariance(sample["P"], s_t, s_r)
        err = sample["error"]

        nees6.append(stable_nees(err, P))
        nees_t.append(stable_nees(err[:3], P[:3, :3]))
        nees_r.append(stable_nees(err[3:], P[3:, 3:]))

        sig_t.append(covariance_sigma(P, slice(0, 3)))
        sig_r.append(covariance_sigma(P, slice(3, 6)))
        err_t.append(float(err[:3].norm()))
        err_r.append(float(err[3:].norm()))

    n6 = torch.tensor(nees6, dtype=torch.float64)
    nt = torch.tensor(nees_t, dtype=torch.float64)
    nr = torch.tensor(nees_r, dtype=torch.float64)

    return {
        "count": len(samples),
        "mean_nees_6d": float(n6.mean()),
        "median_nees_6d": float(n6.median()),
        "mean_nees_t": float(nt.mean()),
        "mean_nees_r": float(nr.mean()),
        "coverage_6d_95": float((n6 <= CHI2_6_95).double().mean()),
        "coverage_6d_99": float((n6 <= CHI2_6_99).double().mean()),
        "coverage_t_95": float((nt <= CHI2_3_95).double().mean()),
        "coverage_t_99": float((nt <= CHI2_3_99).double().mean()),
        "coverage_r_95": float((nr <= CHI2_3_95).double().mean()),
        "coverage_r_99": float((nr <= CHI2_3_99).double().mean()),
        "pearson_t": pearson(sig_t, err_t),
        "spearman_t": spearman(sig_t, err_t),
        "pearson_r": pearson(sig_r, err_r),
        "spearman_r": spearman(sig_r, err_r),
        "median_sigma_t": float(torch.tensor(sig_t).median()),
        "median_sigma_r": float(torch.tensor(sig_r).median()),
        "median_error_t": float(torch.tensor(err_t).median()),
        "median_error_r": float(torch.tensor(err_r).median()),
    }


def format_pct(x):
    return f"{100.0 * x:.2f}%"


def print_metrics(label, metrics):
    print(f"\n{label}")
    print(f"  frames: {metrics['count']}")
    print(
        "  NEES 6D: "
        f"mean={metrics['mean_nees_6d']:.6g}, "
        f"median={metrics['median_nees_6d']:.6g}, "
        f"95%={format_pct(metrics['coverage_6d_95'])}, "
        f"99%={format_pct(metrics['coverage_6d_99'])}"
    )
    print(
        "  NEES trans/rot mean: "
        f"{metrics['mean_nees_t']:.6g} / {metrics['mean_nees_r']:.6g} "
        "(ideal 3 / 3)"
    )
    print(
        "  trans corr: "
        f"Pearson={metrics['pearson_t']:.4f}, "
        f"Spearman={metrics['spearman_t']:.4f}"
    )
    print(
        "  rot   corr: "
        f"Pearson={metrics['pearson_r']:.4f}, "
        f"Spearman={metrics['spearman_r']:.4f}"
    )


def evaluate_split(sequence_samples, scales):
    pooled = [s for samples in sequence_samples.values() for s in samples]

    result = {
        "pooled_raw": summarize(pooled),
        "pooled_calibrated": summarize(
            pooled, scales["s_t"], scales["s_r"]
        ),
        "per_sequence": {},
    }

    for name, samples in sequence_samples.items():
        result["per_sequence"][name] = {
            "raw": summarize(samples),
            "calibrated": summarize(
                samples, scales["s_t"], scales["s_r"]
            ),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Calibration sequences.",
    )
    parser.add_argument(
        "--test",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Held-out sequences.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Cluster covariance block size to calibrate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("m1_calibration.json"),
    )
    args = parser.parse_args()

    train_specs = parse_sequence_specs(args.train)
    test_specs = parse_sequence_specs(args.test)

    overlap = set(n for n, _ in train_specs) & set(n for n, _ in test_specs)
    if overlap:
        raise ValueError(
            "A sequence cannot be both train and test: "
            + ", ".join(sorted(overlap))
        )

    train_sequences = {}
    test_sequences = {}
    covariance_sources = {}

    for name, directory in train_specs:
        samples, source = load_sequence(name, directory, args.block_size)
        train_sequences[name] = samples
        covariance_sources[name] = source

    for name, directory in test_specs:
        samples, source = load_sequence(name, directory, args.block_size)
        test_sequences[name] = samples
        covariance_sources[name] = source

    train_pooled = [
        s for samples in train_sequences.values() for s in samples
    ]
    scales = fit_scales(train_pooled)

    print("=" * 72)
    print("M1 cross-sequence covariance calibration")
    print("=" * 72)
    print(f"block size: {args.block_size}x{args.block_size}")
    print("training sequences: " + ", ".join(train_sequences))
    print("test sequences: " + (", ".join(test_sequences) or "(none)"))
    print(
        "\nfitted calibration:"
        f"\n  s_t={scales['s_t']:.6g}"
        f"  (alpha_t=s_t^2={scales['alpha_t']:.6g})"
        f"\n  s_r={scales['s_r']:.6g}"
        f"  (alpha_r=s_r^2={scales['alpha_r']:.6g})"
    )

    train_eval = evaluate_split(train_sequences, scales)
    print_metrics("TRAIN pooled raw", train_eval["pooled_raw"])
    print_metrics(
        "TRAIN pooled calibrated", train_eval["pooled_calibrated"]
    )
    for name, metrics in train_eval["per_sequence"].items():
        print_metrics(
            f"TRAIN {name} calibrated", metrics["calibrated"]
        )

    test_eval = None
    if test_sequences:
        test_eval = evaluate_split(test_sequences, scales)
        print_metrics("TEST pooled raw", test_eval["pooled_raw"])
        print_metrics(
            "TEST pooled calibrated", test_eval["pooled_calibrated"]
        )
        for name, metrics in test_eval["per_sequence"].items():
            print_metrics(
                f"TEST {name} calibrated", metrics["calibrated"]
            )

    output = {
        "block_size": args.block_size,
        "model": "P_cal = S P_cluster S^T",
        "S_diagonal": [
            scales["s_t"],
            scales["s_t"],
            scales["s_t"],
            scales["s_r"],
            scales["s_r"],
            scales["s_r"],
        ],
        "calibration": scales,
        "train_sequences": {
            name: str(path) for name, path in train_specs
        },
        "test_sequences": {
            name: str(path) for name, path in test_specs
        },
        "covariance_sources": covariance_sources,
        "train_metrics": train_eval,
        "test_metrics": test_eval,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved calibration report: {args.output}")


if __name__ == "__main__":
    main()
