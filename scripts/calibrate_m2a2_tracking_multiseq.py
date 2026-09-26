#!/usr/bin/env python3
"""Cross-sequence calibration for M2-A2 tracking pose covariance.

The input is one or more M2-A2 diagnostic directories containing per-frame
.pt files with:
    P_track_right (or P_track_right_raw),
    T_final,
    T_gt.

The calibration is fitted ONLY from training sequences.

Models:
  raw:
      P_raw = P_track,right

  two-scale baseline:
      P_scale = S P_raw S^T
      S = diag(s_t,s_t,s_t,s_r,s_r,s_r)

  Diag-6 whitened calibration:
      z_i = P_i^(-1/2) e_i
      C_d = diag(mean_i z_i^2)
      P_diag,i = P_i^(1/2) C_d P_i^(1/2)

where
      e_i = Log(T_final^(-1) T_gt)

is the right-invariant final-pose error.

Examples
--------
Explicit train/test split:

python scripts/calibrate_m2a2_tracking_multiseq.py \
  --train seq_a=/run/a/m2a_pose_uncertainty \
          seq_b=/run/b/m2a_pose_uncertainty \
  --test seq_c=/run/c/m2a_pose_uncertainty \
  --output results/m2a2_tracking_calibration.json

Automatic leave-one-sequence-out:

python scripts/calibrate_m2a2_tracking_multiseq.py \
  --loso seq_a=/run/a/m2a_pose_uncertainty \
         seq_b=/run/b/m2a_pose_uncertainty \
         seq_c=/run/c/m2a_pose_uncertainty \
         seq_d=/run/d/m2a_pose_uncertainty \
  --output results/m2a2_tracking_loso.json
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
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty sequence name in {value!r}")
        out.append((name, Path(path).expanduser()))
    return out


def sym(P):
    return 0.5 * (P + P.T)


def stable_eigh(P, floor_rel=1e-10):
    eig, vec = torch.linalg.eigh(sym(P))
    mx = eig.max().clamp_min(1e-18)
    eig = eig.clamp_min(mx * float(floor_rel))
    return eig, vec


def sqrt_and_inv(P, floor_rel=1e-10):
    eig, vec = stable_eigh(P, floor_rel)
    root = (vec * torch.sqrt(eig).unsqueeze(0)) @ vec.T
    invroot = (vec * (1.0 / torch.sqrt(eig)).unsqueeze(0)) @ vec.T
    return sym(root), sym(invroot)


def stable_nees(e, P):
    P = sym(P)
    try:
        return float(e @ torch.linalg.solve(P, e))
    except RuntimeError:
        return float(e @ (torch.linalg.pinv(P) @ e))


def load_sequence(name, directory):
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {directory}")

    samples = []
    source_counts = {}
    for path in files:
        d = torch.load(path, map_location="cpu")
        if "T_final" not in d or "T_gt" not in d:
            continue

        # Newer M2-A2 diagnostics preserve raw tracking covariance explicitly.
        P = d.get("P_track_right_raw", None)
        source = "P_track_right_raw"
        if P is None:
            P = d.get("P_track_right", None)
            source = "P_track_right"
        if P is None:
            continue

        P = sym(P.double())
        if not bool(torch.isfinite(P).all()):
            continue

        T_final = d["T_final"].double()
        T_gt = d["T_gt"].double()
        err = SE3_log(torch.linalg.inv(T_final) @ T_gt)

        samples.append({
            "sequence": name,
            "frame": int(d["frame"]),
            "P": P,
            "error": err,
            "source": source,
        })
        source_counts[source] = source_counts.get(source, 0) + 1

    if not samples:
        raise RuntimeError(
            f"No GT-capable M2-A2 tracking covariance diagnostics in {directory}"
        )
    return samples, source_counts


def pearson(x, y):
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    if x.numel() < 2:
        return float("nan")
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


def fit_two_scale(samples):
    nt, nr = [], []
    for s in samples:
        nt.append(stable_nees(s["error"][:3], s["P"][:3, :3]))
        nr.append(stable_nees(s["error"][3:], s["P"][3:, 3:]))
    mt = float(torch.tensor(nt, dtype=torch.float64).mean())
    mr = float(torch.tensor(nr, dtype=torch.float64).mean())
    alpha_t = max(mt / 3.0, 1e-12)
    alpha_r = max(mr / 3.0, 1e-12)
    return {
        "s_t": math.sqrt(alpha_t),
        "s_r": math.sqrt(alpha_r),
        "alpha_t": alpha_t,
        "alpha_r": alpha_r,
        "train_mean_nees_t_raw": mt,
        "train_mean_nees_r_raw": mr,
    }


def fit_diag6(samples, eig_floor_rel=1e-10):
    z2 = []
    zmean = []
    for s in samples:
        _, invroot = sqrt_and_inv(s["P"], eig_floor_rel)
        z = invroot @ s["error"]
        z2.append(z * z)
        zmean.append(z)
    Z2 = torch.stack(z2)
    Z = torch.stack(zmean)
    diag = Z2.mean(dim=0).clamp_min(1e-12)
    C = torch.diag(diag)
    return {
        "diag": diag,
        "C": C,
        "mean_z": Z.mean(dim=0),
        "std_z": Z.std(dim=0, unbiased=False),
    }


def scale_cov(P, model):
    S = torch.diag(torch.tensor(
        [model["s_t"]] * 3 + [model["s_r"]] * 3,
        dtype=P.dtype,
    ))
    return sym(S @ P @ S)


def diag_cov(P, model, eig_floor_rel=1e-10):
    root, _ = sqrt_and_inv(P, eig_floor_rel)
    C = model["C"].to(dtype=P.dtype)
    return sym(root @ C @ root)


def covariance_for(sample, mode, two_scale=None, diag6=None, eig_floor_rel=1e-10):
    if mode == "raw":
        return sample["P"]
    if mode == "two_scale":
        return scale_cov(sample["P"], two_scale)
    if mode == "diag6":
        return diag_cov(sample["P"], diag6, eig_floor_rel)
    raise ValueError(mode)


def summarize(samples, mode, two_scale=None, diag6=None, eig_floor_rel=1e-10):
    n6, nt, nr = [], [], []
    sigt, sigr, errt, errr = [], [], [], []

    for s in samples:
        P = covariance_for(
            s, mode, two_scale=two_scale, diag6=diag6,
            eig_floor_rel=eig_floor_rel,
        )
        e = s["error"]
        n6.append(stable_nees(e, P))
        nt.append(stable_nees(e[:3], P[:3, :3]))
        nr.append(stable_nees(e[3:], P[3:, 3:]))

        d = torch.diagonal(P).clamp_min(0.0)
        sigt.append(float(torch.sqrt(d[:3].mean())))
        sigr.append(float(torch.sqrt(d[3:].mean())))
        errt.append(float(torch.linalg.norm(e[:3])))
        errr.append(float(torch.linalg.norm(e[3:])))

    n6t = torch.tensor(n6, dtype=torch.float64)
    ntt = torch.tensor(nt, dtype=torch.float64)
    nrt = torch.tensor(nr, dtype=torch.float64)
    return {
        "count": len(samples),
        "mean_nees_6d": float(n6t.mean()),
        "median_nees_6d": float(n6t.median()),
        "mean_nees_t": float(ntt.mean()),
        "mean_nees_r": float(nrt.mean()),
        "coverage_6d_95": float((n6t <= CHI2_6_95).double().mean()),
        "coverage_6d_99": float((n6t <= CHI2_6_99).double().mean()),
        "coverage_t_95": float((ntt <= CHI2_3_95).double().mean()),
        "coverage_t_99": float((ntt <= CHI2_3_99).double().mean()),
        "coverage_r_95": float((nrt <= CHI2_3_95).double().mean()),
        "coverage_r_99": float((nrt <= CHI2_3_99).double().mean()),
        "pearson_t": pearson(sigt, errt),
        "spearman_t": spearman(sigt, errt),
        "pearson_r": pearson(sigr, errr),
        "spearman_r": spearman(sigr, errr),
        "median_sigma_t": float(torch.tensor(sigt).median()),
        "median_sigma_r": float(torch.tensor(sigr).median()),
        "median_error_t": float(torch.tensor(errt).median()),
        "median_error_r": float(torch.tensor(errr).median()),
    }


def print_metrics(label, m):
    print(f"\n{label}")
    print(f"  frames: {m['count']}")
    print(
        "  NEES 6D: "
        f"mean={m['mean_nees_6d']:.6g}, "
        f"median={m['median_nees_6d']:.6g}, "
        f"95%={100*m['coverage_6d_95']:.2f}%, "
        f"99%={100*m['coverage_6d_99']:.2f}%"
    )
    print(
        "  NEES trans/rot mean: "
        f"{m['mean_nees_t']:.6g} / {m['mean_nees_r']:.6g} "
        "(ideal 3 / 3)"
    )
    print(
        "  translation corr: "
        f"Pearson={m['pearson_t']:.4f}, Spearman={m['spearman_t']:.4f}"
    )
    print(
        "  rotation corr: "
        f"Pearson={m['pearson_r']:.4f}, Spearman={m['spearman_r']:.4f}"
    )
    print(
        "  sigma median: "
        f"trans={m['median_sigma_t']:.6g} m, "
        f"rot={m['median_sigma_r']:.6g} rad"
    )


def evaluate(seq_samples, two_scale, diag6, eig_floor_rel):
    pooled = [s for xs in seq_samples.values() for s in xs]

    def one(samples):
        return {
            "raw": summarize(samples, "raw"),
            "two_scale": summarize(
                samples, "two_scale", two_scale=two_scale
            ),
            "diag6": summarize(
                samples, "diag6", diag6=diag6,
                eig_floor_rel=eig_floor_rel,
            ),
        }

    return {
        "pooled": one(pooled),
        "per_sequence": {name: one(xs) for name, xs in seq_samples.items()},
    }


def serialize_calibration(two_scale, diag6, eig_floor_rel):
    return {
        "two_scale": dict(two_scale),
        "diag6": {
            "diag": diag6["diag"].tolist(),
            "C": diag6["C"].tolist(),
            "mean_z": diag6["mean_z"].tolist(),
            "std_z": diag6["std_z"].tolist(),
            "eig_floor_rel": float(eig_floor_rel),
        },
    }


def fit_family(samples, eig_floor_rel):
    return fit_two_scale(samples), fit_diag6(samples, eig_floor_rel)


def macro_aggregate(folds):
    modes = ["raw", "two_scale", "diag6"]
    fields = [
        "mean_nees_6d", "median_nees_6d",
        "coverage_6d_95", "coverage_6d_99",
        "mean_nees_t", "mean_nees_r",
        "pearson_t", "spearman_t",
        "pearson_r", "spearman_r",
    ]
    out = {}
    for mode in modes:
        out[mode] = {}
        for field in fields:
            vals = torch.tensor([
                fold["test_metrics"][mode][field]
                for fold in folds.values()
            ], dtype=torch.float64)
            out[mode][field + "_macro_mean"] = float(vals.mean())
            out[mode][field + "_macro_std"] = float(vals.std(unbiased=False))
            out[mode][field + "_min"] = float(vals.min())
            out[mode][field + "_max"] = float(vals.max())
    return out


def print_macro(agg):
    print("\n" + "=" * 78)
    print("M2-A2 LOSO aggregate summary")
    print("=" * 78)
    for mode in ("raw", "two_scale", "diag6"):
        a = agg[mode]
        print(
            f"{mode:10s} | "
            f"NEES={a['mean_nees_6d_macro_mean']:.4g}"
            f"±{a['mean_nees_6d_macro_std']:.3g} | "
            f"95%={100*a['coverage_6d_95_macro_mean']:.2f}%"
            f"±{100*a['coverage_6d_95_macro_std']:.2f} | "
            f"rho_t={a['spearman_t_macro_mean']:.4f} | "
            f"rho_r={a['spearman_r_macro_mean']:.4f}"
        )


def run_loso(args, specs):
    if len(specs) < 3:
        raise ValueError("LOSO requires at least three sequences.")
    names = [n for n, _ in specs]
    if len(set(names)) != len(names):
        raise ValueError("LOSO sequence names must be unique.")

    sequences, paths, sources = {}, {}, {}
    for name, path in specs:
        xs, src = load_sequence(name, path)
        sequences[name] = xs
        paths[name] = str(path)
        sources[name] = src

    print("=" * 78)
    print("M2-A2 tracking covariance LOSO calibration")
    print("=" * 78)
    print("sequences:", ", ".join(names))

    folds = {}
    for held_out in names:
        train_names = [n for n in names if n != held_out]
        train = [s for n in train_names for s in sequences[n]]
        test = sequences[held_out]

        two_scale, diag6 = fit_family(train, args.eig_floor_rel)
        test_metrics = {
            "raw": summarize(test, "raw"),
            "two_scale": summarize(test, "two_scale", two_scale=two_scale),
            "diag6": summarize(
                test, "diag6", diag6=diag6,
                eig_floor_rel=args.eig_floor_rel,
            ),
        }

        print("\n" + "-" * 78)
        print(f"held out: {held_out}")
        print("train:", ", ".join(train_names))
        print("Diag-6:", [round(float(v), 5) for v in diag6["diag"]])
        print_metrics(f"LOSO {held_out} raw", test_metrics["raw"])
        print_metrics(
            f"LOSO {held_out} two-scale", test_metrics["two_scale"]
        )
        print_metrics(f"LOSO {held_out} diag6", test_metrics["diag6"])

        folds[held_out] = {
            "held_out": held_out,
            "train_sequences": train_names,
            "calibration": serialize_calibration(
                two_scale, diag6, args.eig_floor_rel
            ),
            "test_metrics": test_metrics,
        }

    aggregate = macro_aggregate(folds)
    print_macro(aggregate)

    report = {
        "mode": "loso",
        "model": "m2a2_tracking_covariance",
        "sequence_paths": paths,
        "covariance_sources": sources,
        "folds": folds,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {args.output}")


def run_split(args, train_specs, test_specs):
    overlap = set(n for n, _ in train_specs) & set(n for n, _ in test_specs)
    if overlap:
        raise ValueError(
            "A sequence cannot be both train and test: "
            + ", ".join(sorted(overlap))
        )

    train_sequences, test_sequences = {}, {}
    sources = {}
    for name, path in train_specs:
        xs, src = load_sequence(name, path)
        train_sequences[name] = xs
        sources[name] = src
    for name, path in test_specs:
        xs, src = load_sequence(name, path)
        test_sequences[name] = xs
        sources[name] = src

    train_pooled = [s for xs in train_sequences.values() for s in xs]
    two_scale, diag6 = fit_family(train_pooled, args.eig_floor_rel)

    print("=" * 78)
    print("M2-A2 tracking covariance cross-sequence calibration")
    print("=" * 78)
    print("train:", ", ".join(train_sequences))
    print("test:", ", ".join(test_sequences) or "(none)")
    print(
        "two-scale: "
        f"s_t={two_scale['s_t']:.6g}, s_r={two_scale['s_r']:.6g}"
    )
    print(
        "Diag-6: "
        + str([round(float(v), 6) for v in diag6["diag"]])
    )

    train_eval = evaluate(
        train_sequences, two_scale, diag6, args.eig_floor_rel
    )
    print_metrics("TRAIN pooled raw", train_eval["pooled"]["raw"])
    print_metrics(
        "TRAIN pooled two-scale", train_eval["pooled"]["two_scale"]
    )
    print_metrics("TRAIN pooled diag6", train_eval["pooled"]["diag6"])

    for name, ms in train_eval["per_sequence"].items():
        print_metrics(f"TRAIN {name} diag6", ms["diag6"])

    test_eval = None
    if test_sequences:
        test_eval = evaluate(
            test_sequences, two_scale, diag6, args.eig_floor_rel
        )
        print_metrics("TEST pooled raw", test_eval["pooled"]["raw"])
        print_metrics(
            "TEST pooled two-scale", test_eval["pooled"]["two_scale"]
        )
        print_metrics("TEST pooled diag6", test_eval["pooled"]["diag6"])
        for name, ms in test_eval["per_sequence"].items():
            print_metrics(f"TEST {name} diag6", ms["diag6"])

    report = {
        "mode": "split",
        "model": "m2a2_tracking_covariance",
        "train_sequences": {
            n: str(p) for n, p in train_specs
        },
        "test_sequences": {
            n: str(p) for n, p in test_specs
        },
        "covariance_sources": sources,
        "calibration": serialize_calibration(
            two_scale, diag6, args.eig_floor_rel
        ),
        "train_metrics": train_eval,
        "test_metrics": test_eval,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", default=[], metavar="NAME=PATH")
    p.add_argument("--test", nargs="*", default=[], metavar="NAME=PATH")
    p.add_argument("--loso", nargs="+", default=[], metavar="NAME=PATH")
    p.add_argument("--eig-floor-rel", type=float, default=1e-10)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/m2a2_tracking_calibration.json"),
    )
    args = p.parse_args()

    train_specs = parse_specs(args.train)
    test_specs = parse_specs(args.test)
    loso_specs = parse_specs(args.loso)

    if loso_specs:
        if train_specs or test_specs:
            raise ValueError("--loso cannot be combined with --train/--test")
        run_loso(args, loso_specs)
        return

    if not train_specs:
        raise ValueError("Provide --train ... or --loso ...")
    run_split(args, train_specs, test_specs)


if __name__ == "__main__":
    main()
