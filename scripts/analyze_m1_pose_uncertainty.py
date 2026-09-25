#!/usr/bin/env python3
"""Analyze M1 pose uncertainty, covariance variants, and pose conventions."""

import argparse
import math
from pathlib import Path
import torch


def scalar(x):
    return float(x.item()) if torch.is_tensor(x) else float(x)


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
    return I + ((1 - torch.cos(a)) / a2) * W + ((a - torch.sin(a)) / (a2 * a)) * (W @ W)


def se3_log(T):
    w = so3_log(T[:3, :3])
    V = left_jacobian_so3(w)
    t = T[:3, 3]
    try:
        rho = torch.linalg.solve(V, t)
    except RuntimeError:
        rho = torch.linalg.pinv(V) @ t
    return torch.cat([rho, w])


def pearson(x, y):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def rankdata(x):
    x = torch.as_tensor(x, dtype=torch.float64)
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
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def stats(name, xs):
    x = torch.as_tensor(xs, dtype=torch.float64)
    x = x[torch.isfinite(x)]
    if not len(x):
        print(f"{name:>22s}: no finite samples")
        return
    print(
        f"{name:>22s}: median={x.median().item():.6g}, "
        f"mean={x.mean().item():.6g}, max={x.max().item():.6g}"
    )


def rms_sigma(P, sl):
    d = torch.diagonal(P)[sl].clamp_min(0.0)
    return float(torch.sqrt(d.mean()))


def nees(err, P):
    P = 0.5 * (P + P.T)
    try:
        return float(err @ torch.linalg.solve(P, err))
    except RuntimeError:
        return float(err @ (torch.linalg.pinv(P) @ err))


def summarize_cov(rows, label, pkey):
    # Per-covariance values are stored with expanded keys such as
    # "hess_sig_t", "cluster_sig_t", and "main_sig_t". Checking for the
    # bare prefix (e.g. "hess") incorrectly filters every row out.
    sig_key = pkey + "_sig_t"
    nees_key = pkey + "_nees"
    valid = [r for r in rows if sig_key in r and nees_key in r]
    if not valid:
        print(f"\n=== {label} covariance ===")
        print("no valid covariance diagnostics found")
        return

    sig_t = [r[pkey + "_sig_t"] for r in valid]
    sig_r = [r[pkey + "_sig_r"] for r in valid]
    err_t = [r["err_param_t"] for r in valid]
    err_r = [r["err_param_r"] for r in valid]
    ns = [r[pkey + "_nees"] for r in valid]

    print(f"\n=== {label} covariance ===")
    stats("sigma_t_rms", sig_t)
    stats("sigma_r_rms", sig_r)
    stats("NEES_6D", ns)
    print(
        "error-uncertainty correlation:\n"
        f"  translation Pearson={pearson(sig_t, err_t):.4f}, "
        f"Spearman={spearman(sig_t, err_t):.4f}\n"
        f"  rotation    Pearson={pearson(sig_r, err_r):.4f}, "
        f"Spearman={spearman(sig_r, err_r):.4f}"
    )

    nst = torch.tensor(ns, dtype=torch.float64)
    print(
        f"NEES mean={nst.mean().item():.6g} (ideal 6), "
        f"95% coverage={(nst <= 12.591587).double().mean().item():.3%}, "
        f"99% coverage={(nst <= 16.811894).double().mean().item():.3%}"
    )

    order = sorted(valid, key=lambda r: r[pkey + "_sig_t"])
    print("translation error by uncertainty quintile:")
    N = len(order)
    for b in range(5):
        chunk = order[b * N // 5:(b + 1) * N // 5]
        if not chunk:
            continue
        u = torch.tensor([r[pkey + "_sig_t"] for r in chunk])
        e = torch.tensor([r["err_param_t"] for r in chunk])
        print(
            f"  Q{b+1}: unc_med={u.median().item():.4e}, "
            f"err_med={e.median().item():.4e}, err_mean={e.mean().item():.4e}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--psd-tol", type=float, default=1e-9)
    args = ap.parse_args()

    files = sorted(args.directory.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files in {args.directory}")

    rows = []
    bad = {"finite": 0, "sym": 0, "psd": 0}

    for path in files:
        d = torch.load(path, map_location="cpu")
        Pmain = d["P_xi_raw"].double()
        Ps = {
            "main": Pmain,
            "hess": d.get("P_xi_hessian", Pmain).double(),
            "cluster": d.get("P_xi_cluster", Pmain).double(),
        }

        eig = torch.linalg.eigvalsh(0.5 * (Pmain + Pmain.T))
        bad["finite"] += int(not bool(torch.isfinite(Pmain).all()))
        bad["sym"] += int(not bool(torch.allclose(Pmain, Pmain.T, atol=1e-8, rtol=1e-6)))
        bad["psd"] += int(bool(eig.min() < -args.psd_tol))

        r = {
            "frame": int(d["frame"]),
            "condition": scalar(d["condition"]),
            "fb": scalar(d["fb_error_median_px"]),
            "maha": scalar(d["maha_median"]),
            "n": int(d["num_pixels"]),
            "mode": d.get("covariance_mode", "legacy"),
            "clusters": int(d.get("cluster_count", 0)),
            "block": int(d.get("cluster_block_size", 0)),
        }

        if "T_rel_gt" in d:
            Tgt = d["T_rel_gt"].double()
            Test = d["T_rel_raw"].double()
            xi_est = d["xi_raw"].double()
            xi_gt = se3_log(Tgt)
            xi_gt_inv = se3_log(torch.linalg.inv(Tgt))

            # Parameter-space error is the quantity directly associated with
            # the covariance of the fitted xi.
            e_param = xi_est - xi_gt
            e_param_inv = xi_est - xi_gt_inv

            # Group error is logged as a convention sanity check.
            e_group = se3_log(torch.linalg.inv(Tgt) @ Test)

            r.update({
                "err_param_t": float(e_param[:3].norm()),
                "err_param_r": float(e_param[3:].norm()),
                "err_param_inv_t": float(e_param_inv[:3].norm()),
                "err_param_inv_r": float(e_param_inv[3:].norm()),
                "err_group_t": float(e_group[:3].norm()),
                "err_group_r": float(e_group[3:].norm()),
            })

            for key, P in Ps.items():
                P = 0.5 * (P + P.T)
                r[key + "_sig_t"] = rms_sigma(P, slice(0, 3))
                r[key + "_sig_r"] = rms_sigma(P, slice(3, 6))
                r[key + "_nees"] = nees(e_param, P)

        rows.append(r)

    print(f"frames: {len(rows)}")
    print(
        f"numerical checks (main covariance): nonfinite={bad['finite']}, "
        f"nonsymmetric={bad['sym']}, nonPSD={bad['psd']}"
    )
    stats("condition", [r["condition"] for r in rows])
    stats("fb_error_px", [r["fb"] for r in rows])
    stats("maha_median", [r["maha"] for r in rows])
    stats("cluster_count", [r["clusters"] for r in rows])

    gt = [r for r in rows if "err_param_t" in r]
    if not gt:
        print("No GT in these logs. Re-run the upgraded M1 branch.")
        return

    print("\n=== Pose convention audit ===")
    stats("param trans error", [r["err_param_t"] for r in gt])
    stats("param rot error", [r["err_param_r"] for r in gt])
    stats("inverse trans error", [r["err_param_inv_t"] for r in gt])
    stats("inverse rot error", [r["err_param_inv_r"] for r in gt])
    stats("group trans error", [r["err_group_t"] for r in gt])
    stats("group rot error", [r["err_group_r"] for r in gt])

    med_direct = torch.tensor([r["err_param_t"] + r["err_param_r"] for r in gt]).median()
    med_inv = torch.tensor([r["err_param_inv_t"] + r["err_param_inv_r"] for r in gt]).median()
    print(
        "convention check: "
        + ("stored T_rel_gt direction is preferred" if med_direct < med_inv
           else "WARNING: inverse T_rel_gt fits xi better; inspect pose convention")
    )

    summarize_cov(gt, "naive Hessian", "hess")
    summarize_cov(gt, "cluster-robust", "cluster")
    summarize_cov(gt, "selected/main", "main")


if __name__ == "__main__":
    main()
