#!/usr/bin/env python3
"""Analyze M2-A absolute right-invariant pose covariance diagnostics.

Example:
    python scripts/analyze_m2a_pose_uncertainty.py \
        results/m2a_pose_uncertainty
"""

import argparse
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


def stable_nees(e, P):
    P = 0.5 * (P + P.T)
    try:
        return float(e @ torch.linalg.solve(P, e))
    except RuntimeError:
        return float(e @ (torch.linalg.pinv(P) @ e))


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


def summarize(rows, key, label):
    valid = [r for r in rows if key in r]
    if not valid:
        print(f"\n=== {label} ===\nno valid rows")
        return

    nees6, neest, neesr = [], [], []
    sigt, sigr, errt, errr = [], [], [], []
    nonfinite = nonsym = nonpsd = 0

    for r in valid:
        P = r[key]
        if not bool(torch.isfinite(P).all()):
            nonfinite += 1
            continue
        if float(torch.max(torch.abs(P - P.T))) > 1e-8:
            nonsym += 1
        eig = torch.linalg.eigvalsh(0.5 * (P + P.T))
        if float(eig.min()) < -1e-10:
            nonpsd += 1

        e = r["err"]
        nees6.append(stable_nees(e, P))
        neest.append(stable_nees(e[:3], P[:3, :3]))
        neesr.append(stable_nees(e[3:], P[3:, 3:]))
        d = torch.diagonal(P).clamp_min(0.0)
        sigt.append(float(torch.sqrt(d[:3].mean())))
        sigr.append(float(torch.sqrt(d[3:].mean())))
        errt.append(float(torch.linalg.norm(e[:3])))
        errr.append(float(torch.linalg.norm(e[3:])))

    n6 = torch.tensor(nees6, dtype=torch.float64)
    nt = torch.tensor(neest, dtype=torch.float64)
    nr = torch.tensor(neesr, dtype=torch.float64)

    print(f"\n=== {label} ===")
    print(f"frames: {len(nees6)}")
    print(
        "numerical checks: "
        f"nonfinite={nonfinite}, nonsymmetric={nonsym}, nonPSD={nonpsd}"
    )
    print(
        "NEES 6D: "
        f"median={float(n6.median()):.6g}, "
        f"mean={float(n6.mean()):.6g}, "
        f"95%={100*float((n6 <= CHI2_6_95).double().mean()):.2f}%, "
        f"99%={100*float((n6 <= CHI2_6_99).double().mean()):.2f}%"
    )
    print(
        "NEES trans/rot mean: "
        f"{float(nt.mean()):.6g} / {float(nr.mean()):.6g} "
        "(ideal 3 / 3)"
    )
    print(
        "translation corr: "
        f"Pearson={pearson(sigt, errt):.4f}, "
        f"Spearman={spearman(sigt, errt):.4f}"
    )
    print(
        "rotation corr: "
        f"Pearson={pearson(sigr, errr):.4f}, "
        f"Spearman={spearman(sigr, errr):.4f}"
    )
    print(
        "sigma rms median: "
        f"trans={float(torch.tensor(sigt).median()):.6g} m, "
        f"rot={float(torch.tensor(sigr).median()):.6g} rad"
    )


def summarize_tracking_update(rows):
    valid = [
        r for r in rows
        if "err_prior" in r and "err_final" in r
    ]
    if not valid:
        print("\n=== M2-A2 tracking prior/posterior audit ===")
        print("no frames with T_motion_prior")
        return

    prior_t, final_t = [], []
    prior_r, final_r = [], []
    gain_t, gain_r = [], []
    ratio_t, ratio_r = [], []

    for r in valid:
        ep = r["err_prior"]
        ef = r["err_final"]

        pt = float(torch.linalg.norm(ep[:3]))
        ft = float(torch.linalg.norm(ef[:3]))
        pr = float(torch.linalg.norm(ep[3:]))
        fr = float(torch.linalg.norm(ef[3:]))

        prior_t.append(pt)
        final_t.append(ft)
        prior_r.append(pr)
        final_r.append(fr)

        gain_t.append(pt - ft)
        gain_r.append(pr - fr)
        ratio_t.append(ft / max(pt, 1e-12))
        ratio_r.append(fr / max(pr, 1e-12))

    def stats(x):
        t = torch.tensor(x, dtype=torch.float64)
        return float(t.median()), float(t.mean()), float(t.max())

    p_tm, p_tmean, _ = stats(prior_t)
    f_tm, f_tmean, _ = stats(final_t)
    p_rm, p_rmean, _ = stats(prior_r)
    f_rm, f_rmean, _ = stats(final_r)
    g_tm, g_tmean, _ = stats(gain_t)
    g_rm, g_rmean, _ = stats(gain_r)
    rt_med, rt_mean, _ = stats(ratio_t)
    rr_med, rr_mean, _ = stats(ratio_r)

    improved_t = sum(g > 0 for g in gain_t) / len(gain_t)
    improved_r = sum(g > 0 for g in gain_r) / len(gain_r)

    print("\n=== M2-A2 tracking prior/posterior audit ===")
    print(f"frames: {len(valid)}")
    print(
        "translation error prior -> final: "
        f"median {p_tm:.6g} -> {f_tm:.6g} m, "
        f"mean {p_tmean:.6g} -> {f_tmean:.6g} m"
    )
    print(
        "rotation error prior -> final: "
        f"median {p_rm:.6g} -> {f_rm:.6g} rad, "
        f"mean {p_rmean:.6g} -> {f_rmean:.6g} rad"
    )
    print(
        "tracking error reduction: "
        f"translation median={g_tm:.6g}, mean={g_tmean:.6g}, "
        f"improved={100.0*improved_t:.2f}%"
    )
    print(
        "tracking error reduction: "
        f"rotation median={g_rm:.6g}, mean={g_rmean:.6g}, "
        f"improved={100.0*improved_r:.2f}%"
    )
    print(
        "final/prior error ratio: "
        f"translation median={rt_med:.4f}, mean={rt_mean:.4f}; "
        f"rotation median={rr_med:.4f}, mean={rr_mean:.4f}"
    )
    print(
        "prior-vs-final error correlation: "
        f"translation Pearson={pearson(prior_t, final_t):.4f}, "
        f"Spearman={spearman(prior_t, final_t):.4f}"
    )
    print(
        "prior-vs-final error correlation: "
        f"rotation Pearson={pearson(prior_r, final_r):.4f}, "
        f"Spearman={spearman(prior_r, final_r):.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    files = sorted(args.directory.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {args.directory}")

    rows = []
    track_dt, track_dr = [], []
    sources = {}

    for path in files:
        d = torch.load(path, map_location="cpu")
        if "T_final" not in d or "T_gt" not in d:
            continue
        T_est = d["T_final"].double()
        T_gt = d["T_gt"].double()
        err = SE3_log(torch.linalg.inv(T_est) @ T_gt)

        row = {
            "frame": int(d["frame"]),
            "err": err,
            "err_final": err,
            "P_abs": d["P_abs_right"].double(),
            "P_abs_raw": d["P_abs_right_raw"].double(),
        }

        if d.get("T_motion_prior", None) is not None:
            T_prior = d["T_motion_prior"].double()
            row["err_prior"] = SE3_log(
                torch.linalg.inv(T_prior) @ T_gt
            )
        rows.append(row)

        source = str(d.get("source", "unknown"))
        sources[source] = sources.get(source, 0) + 1

        dt = float(d.get("tracking_correction_translation_m", float("nan")))
        dr = float(d.get("tracking_correction_rotation_rad", float("nan")))
        if torch.isfinite(torch.tensor(dt)):
            track_dt.append(dt)
        if torch.isfinite(torch.tensor(dr)):
            track_dr.append(dr)

    print("=== M2-A absolute pose covariance audit ===")
    print(f"diagnostic frames: {len(rows)}")
    print("sources:", sources)

    summarize(rows, "P_abs_raw", "raw recursively propagated covariance")
    summarize(rows, "P_abs", "selected recursively propagated covariance")
    summarize_tracking_update(rows)

    if track_dt:
        x = torch.tensor(track_dt)
        print(
            "\ntracking correction translation: "
            f"median={float(x.median()):.6g}, "
            f"mean={float(x.mean()):.6g}, max={float(x.max()):.6g} m"
        )
    if track_dr:
        x = torch.tensor(track_dr)
        print(
            "tracking correction rotation: "
            f"median={float(x.median()):.6g}, "
            f"mean={float(x.mean()):.6g}, max={float(x.max()):.6g} rad"
        )


if __name__ == "__main__":
    main()
