#!/usr/bin/env python3
"""Analyze M1 relative-pose uncertainty and, when available, GT calibration.

Usage:
    python scripts/analyze_m1_pose_uncertainty.py /path/to/run/m1_pose_uncertainty

New M1 logs contain T_rel_gt and support:
- numerical covariance checks,
- relative translation / rotation errors,
- Pearson and Spearman error-uncertainty correlations,
- 6-DoF NEES and chi-square coverage summaries,
- uncertainty-bin calibration tables.

Older logs without T_rel_gt are still supported for numerical diagnostics.
"""

import argparse
import math
from pathlib import Path

import torch


def _scalar(x):
    if torch.is_tensor(x):
        return float(x.item())
    return float(x)


def _skew(w):
    W = torch.zeros((3, 3), dtype=w.dtype)
    W[0, 1], W[0, 2] = -w[2], w[1]
    W[1, 0], W[1, 2] = w[2], -w[0]
    W[2, 0], W[2, 1] = -w[1], w[0]
    return W


def _so3_log(R):
    cos_theta = torch.clamp((torch.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta < 1e-8:
        return 0.5 * torch.tensor(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=R.dtype,
        )

    vee = torch.tensor(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=R.dtype,
    )
    return theta / (2.0 * torch.sin(theta)) * vee


def _left_jacobian_so3(w):
    theta = w.norm()
    W = _skew(w)
    I = torch.eye(3, dtype=w.dtype)
    if theta < 1e-8:
        return I + 0.5 * W + (1.0 / 6.0) * (W @ W)
    theta2 = theta * theta
    A = (1.0 - torch.cos(theta)) / theta2
    B = (theta - torch.sin(theta)) / (theta2 * theta)
    return I + A * W + B * (W @ W)


def _se3_log(T):
    """Inverse of the repository's SE3_exp convention [rho, theta]."""
    R = T[:3, :3]
    t = T[:3, 3]
    w = _so3_log(R)
    V = _left_jacobian_so3(w)
    try:
        rho = torch.linalg.solve(V, t)
    except RuntimeError:
        rho = torch.linalg.pinv(V) @ t
    return torch.cat([rho, w])


def _relative_error(T_est, T_gt):
    """Right-invariant relative-transform error: Log(T_gt^{-1} T_est)."""
    return _se3_log(torch.linalg.inv(T_gt) @ T_est)


def _pearson(x, y):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum())
    if den <= 0:
        return float("nan")
    return float((x * y).sum() / den)


def _rankdata(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    sorted_x = x[order]
    n = x.numel()
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _stats(name, x):
    x = torch.as_tensor(x, dtype=torch.float64)
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        print(f"{name:>18s}: no finite samples")
        return
    print(
        f"{name:>18s}: "
        f"median={finite.median().item():.6g}, "
        f"mean={finite.mean().item():.6g}, "
        f"max={finite.max().item():.6g}"
    )


def _print_calibration_bins(rows, uncertainty_key, error_key, bins=5):
    rows = [
        r for r in rows
        if math.isfinite(r[uncertainty_key]) and math.isfinite(r[error_key])
    ]
    if len(rows) < bins:
        return

    rows = sorted(rows, key=lambda r: r[uncertainty_key])
    print(f"\n{error_key} by {uncertainty_key} quantile:")
    n = len(rows)
    for b in range(bins):
        lo = b * n // bins
        hi = (b + 1) * n // bins
        chunk = rows[lo:hi]
        u = torch.tensor([r[uncertainty_key] for r in chunk], dtype=torch.float64)
        e = torch.tensor([r[error_key] for r in chunk], dtype=torch.float64)
        print(
            f"  Q{b+1}: N={len(chunk):4d} "
            f"unc_med={u.median().item():.4e} "
            f"err_med={e.median().item():.4e} "
            f"err_mean={e.mean().item():.4e}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--psd-tol", type=float, default=1e-9)
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    files = sorted(args.directory.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt diagnostics found in {args.directory}")

    rows = []
    bad_finite = bad_symmetry = bad_psd = 0
    gt_count = 0

    for path in files:
        data = torch.load(path, map_location="cpu")
        P = data["P_xi_raw"].double()
        P_sym = 0.5 * (P + P.T)

        finite = bool(torch.isfinite(P).all())
        symmetric = bool(torch.allclose(P, P.T, atol=1e-8, rtol=1e-6))
        eig = torch.linalg.eigvalsh(P_sym)
        psd = bool(eig.min() >= -args.psd_tol)

        bad_finite += int(not finite)
        bad_symmetry += int(not symmetric)
        bad_psd += int(not psd)

        sig_t = data["sigma_trans_m"].double()
        sig_r = data["sigma_rot_rad"].double()

        row = {
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
            "has_gt": "T_rel_gt" in data,
        }

        if "T_rel_gt" in data:
            gt_count += 1
            T_gt = data["T_rel_gt"].double()
            T_raw = data["T_rel_raw"].double()
            T_applied = data.get("T_rel_applied", data["T_rel_raw"]).double()

            err_raw = _relative_error(T_raw, T_gt)
            err_applied = _relative_error(T_applied, T_gt)

            row["trans_error_raw"] = float(err_raw[:3].norm())
            row["rot_error_raw"] = float(err_raw[3:].norm())
            row["trans_error_applied"] = float(err_applied[:3].norm())
            row["rot_error_applied"] = float(err_applied[3:].norm())

            try:
                nees = err_raw @ torch.linalg.solve(P_sym, err_raw)
            except RuntimeError:
                nees = err_raw @ (torch.linalg.pinv(P_sym) @ err_raw)
            row["nees"] = float(nees)

            P_t = P_sym[:3, :3]
            P_r = P_sym[3:, 3:]
            try:
                row["nees_trans"] = float(err_raw[:3] @ torch.linalg.solve(P_t, err_raw[:3]))
            except RuntimeError:
                row["nees_trans"] = float(err_raw[:3] @ (torch.linalg.pinv(P_t) @ err_raw[:3]))
            try:
                row["nees_rot"] = float(err_raw[3:] @ torch.linalg.solve(P_r, err_raw[3:]))
            except RuntimeError:
                row["nees_rot"] = float(err_raw[3:] @ (torch.linalg.pinv(P_r) @ err_raw[3:]))

        rows.append(row)

    print(f"frames: {len(rows)}")
    print(
        "numerical checks: "
        f"nonfinite={bad_finite}, nonsymmetric={bad_symmetry}, nonPSD={bad_psd}"
    )

    finite_rows = [r for r in rows if math.isfinite(r["condition"]) and math.isfinite(r["trans_rms"])]
    if finite_rows:
        _stats("sigma_t_rms", [r["trans_rms"] for r in finite_rows])
        _stats("sigma_r_rms", [r["rot_rms"] for r in finite_rows])
        _stats("condition", [r["condition"] for r in finite_rows])
        _stats("kappa", [r["kappa"] for r in finite_rows])
        _stats("fb_error_px", [r["fb"] for r in finite_rows])
        _stats("maha_median", [r["maha"] for r in finite_rows])

    print("\nTop frames by translation uncertainty:")
    for r in sorted(rows, key=lambda x: x["trans_rms"], reverse=True)[:10]:
        suffix = ""
        if r["has_gt"]:
            suffix = (
                f" et={r['trans_error_raw']:.3e}"
                f" er={r['rot_error_raw']:.3e}"
                f" nees={r['nees']:.3e}"
            )
        print(
            f"frame={r['frame']:6d} "
            f"sigma_t={r['trans_rms']:.4e} "
            f"sigma_r={r['rot_rms']:.4e} "
            f"cond={r['condition']:.3e} "
            f"kappa={r['kappa']:.3e} "
            f"fb={r['fb']:.3f} "
            f"N={r['n']}{suffix}"
        )

    if gt_count == 0:
        print(
            "\nGround-truth relative poses are not present in these diagnostics. "
            "Re-run the current M1 branch to enable error correlation and NEES."
        )
        return

    gt_rows = [r for r in rows if r["has_gt"] and math.isfinite(r["nees"])]
    print(f"\nGT calibration frames: {len(gt_rows)}/{len(rows)}")
    _stats("trans_err_raw_m", [r["trans_error_raw"] for r in gt_rows])
    _stats("rot_err_raw_rad", [r["rot_error_raw"] for r in gt_rows])
    _stats("trans_err_applied", [r["trans_error_applied"] for r in gt_rows])
    _stats("rot_err_applied", [r["rot_error_applied"] for r in gt_rows])
    _stats("NEES_6D", [r["nees"] for r in gt_rows])
    _stats("NEES_trans_3D", [r["nees_trans"] for r in gt_rows])
    _stats("NEES_rot_3D", [r["nees_rot"] for r in gt_rows])

    sigma_t = [r["trans_rms"] for r in gt_rows]
    sigma_r = [r["rot_rms"] for r in gt_rows]
    err_t = [r["trans_error_raw"] for r in gt_rows]
    err_r = [r["rot_error_raw"] for r in gt_rows]

    print("\nError-uncertainty correlation (raw probabilistic refit):")
    print(
        f"  translation: Pearson={_pearson(sigma_t, err_t):.4f}, "
        f"Spearman={_spearman(sigma_t, err_t):.4f}"
    )
    print(
        f"  rotation:    Pearson={_pearson(sigma_r, err_r):.4f}, "
        f"Spearman={_spearman(sigma_r, err_r):.4f}"
    )
    print(
        f"  fb->trans error: Pearson={_pearson([r['fb'] for r in gt_rows], err_t):.4f}, "
        f"Spearman={_spearman([r['fb'] for r in gt_rows], err_t):.4f}"
    )

    chi2_6_95 = 12.591587
    chi2_6_99 = 16.811894
    chi2_3_95 = 7.814728
    nees = torch.tensor([r["nees"] for r in gt_rows], dtype=torch.float64)
    nees_t = torch.tensor([r["nees_trans"] for r in gt_rows], dtype=torch.float64)
    nees_r = torch.tensor([r["nees_rot"] for r in gt_rows], dtype=torch.float64)

    print("\nNEES calibration summary:")
    print("  ideal mean NEES (6 DoF): 6.0")
    print(f"  empirical mean NEES:      {nees.mean().item():.6g}")
    print(
        f"  coverage NEES <= chi2_6(0.95): "
        f"{(nees <= chi2_6_95).double().mean().item():.3%} (ideal ~95%)"
    )
    print(
        f"  coverage NEES <= chi2_6(0.99): "
        f"{(nees <= chi2_6_99).double().mean().item():.3%} (ideal ~99%)"
    )
    print(
        f"  trans 3D coverage <= chi2_3(0.95): "
        f"{(nees_t <= chi2_3_95).double().mean().item():.3%}"
    )
    print(
        f"  rot   3D coverage <= chi2_3(0.95): "
        f"{(nees_r <= chi2_3_95).double().mean().item():.3%}"
    )

    _print_calibration_bins(gt_rows, "trans_rms", "trans_error_raw", bins=args.bins)
    _print_calibration_bins(gt_rows, "rot_rms", "rot_error_raw", bins=args.bins)


if __name__ == "__main__":
    main()
