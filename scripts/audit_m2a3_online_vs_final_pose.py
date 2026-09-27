#!/usr/bin/env python3
"""Audit online M2 tracking poses against final backend-refined poses.

M2-A2 diagnostics are saved inside frontend tracking(), before later backend
updates may refine camera poses.  Flow4DGS final ATE, in contrast, is evaluated
from the final camera states.  This script makes that distinction explicit.

For each run it compares:
  * online pose: T_final stored in m2a_pose_uncertainty/*.pt (world->camera)
  * final pose:  pose.txt written by eval_ate() (camera->world)
  * GT pose:     T_gt stored in the same M2 diagnostic (world->camera)

It reports raw camera-center / rotation errors, online-to-final corrections,
and rigid-aligned translation RMSE for online and final trajectories.

Diagnostic only; no SLAM runtime behavior is changed.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def parse_specs(values):
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=RUN_DIR, got {value!r}")
        name, path = value.split("=", 1)
        out.append((name.strip(), Path(path).expanduser()))
    return out


def load_pose_txt(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            if not vals:
                continue
            if len(vals) != 4:
                raise ValueError(f"Expected 4 columns in {path}, got: {line}")
            rows.append(vals)
    if len(rows) % 4 != 0:
        raise ValueError(
            f"{path} has {len(rows)} nonempty rows; expected a multiple of 4"
        )
    poses = np.asarray(rows, dtype=np.float64).reshape(-1, 4, 4)
    return poses


def rotation_angle(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(c))


def rigid_align_rmse(est_xyz, gt_xyz):
    """SE(3) Horn/Kabsch alignment, no scale."""
    est_mean = est_xyz.mean(axis=0)
    gt_mean = gt_xyz.mean(axis=0)
    X = est_xyz - est_mean
    Y = gt_xyz - gt_mean
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = gt_mean - R @ est_mean
    aligned = (R @ est_xyz.T).T + t
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    return float(np.sqrt(np.mean(err * err))), R, t, err


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
    }


def audit_run(name, run_dir):
    m2_dir = run_dir / "m2a_pose_uncertainty"
    pose_path = run_dir / "pose.txt"
    files = sorted(m2_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No M2 diagnostics: {m2_dir}")
    if not pose_path.exists():
        raise FileNotFoundError(
            f"Missing final pose file: {pose_path}. "
            "Use the run directory that contains pose.txt."
        )

    final_c2w = load_pose_txt(pose_path)

    frames = []
    online_c2w = []
    gt_c2w = []
    final_selected = []

    for p in files:
        d = torch.load(p, map_location="cpu")
        frame = int(d.get("frame", int(p.stem)))
        T_online = d.get("T_final")
        T_gt = d.get("T_gt")
        if T_online is None or T_gt is None:
            continue
        if frame < 0 or frame >= len(final_c2w):
            continue

        T_online = T_online.double().numpy()
        T_gt = T_gt.double().numpy()
        if not np.isfinite(T_online).all() or not np.isfinite(T_gt).all():
            continue

        frames.append(frame)
        online_c2w.append(np.linalg.inv(T_online))
        gt_c2w.append(np.linalg.inv(T_gt))
        final_selected.append(final_c2w[frame])

    online_c2w = np.asarray(online_c2w)
    gt_c2w = np.asarray(gt_c2w)
    final_selected = np.asarray(final_selected)
    if len(frames) == 0:
        raise RuntimeError(f"No matched frames for {name}")

    C_on = online_c2w[:, :3, 3]
    C_fin = final_selected[:, :3, 3]
    C_gt = gt_c2w[:, :3, 3]

    online_pos_raw = np.linalg.norm(C_on - C_gt, axis=1)
    final_pos_raw = np.linalg.norm(C_fin - C_gt, axis=1)
    online_final_pos = np.linalg.norm(C_on - C_fin, axis=1)

    online_rot_raw = []
    final_rot_raw = []
    online_final_rot = []
    for Ton, Tfin, Tgt in zip(online_c2w, final_selected, gt_c2w):
        online_rot_raw.append(
            rotation_angle(Ton[:3, :3].T @ Tgt[:3, :3])
        )
        final_rot_raw.append(
            rotation_angle(Tfin[:3, :3].T @ Tgt[:3, :3])
        )
        online_final_rot.append(
            rotation_angle(Ton[:3, :3].T @ Tfin[:3, :3])
        )

    online_aligned_rmse, _, _, online_aligned_err = rigid_align_rmse(
        C_on, C_gt
    )
    final_aligned_rmse, _, _, final_aligned_err = rigid_align_rmse(
        C_fin, C_gt
    )

    return {
        "name": name,
        "run_dir": str(run_dir),
        "matched_frames": len(frames),
        "frame_min": min(frames),
        "frame_max": max(frames),
        "pose_txt_frames": int(len(final_c2w)),
        "online_camera_center_error_raw_m": stats(online_pos_raw),
        "final_camera_center_error_raw_m": stats(final_pos_raw),
        "online_to_final_camera_center_correction_m": stats(online_final_pos),
        "online_rotation_error_raw_rad": stats(online_rot_raw),
        "final_rotation_error_raw_rad": stats(final_rot_raw),
        "online_to_final_rotation_correction_rad": stats(online_final_rot),
        "online_aligned_translation_rmse_m": online_aligned_rmse,
        "final_aligned_translation_rmse_m": final_aligned_rmse,
        "online_aligned_translation_error_m": stats(online_aligned_err),
        "final_aligned_translation_error_m": stats(final_aligned_err),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", nargs="+", required=True, metavar="NAME=RUN_DIR"
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("results/m2a3_online_vs_final_pose.json"),
    )
    args = p.parse_args()

    report = {}
    print("=" * 82)
    print("M2-A3 online tracking pose vs final backend-refined pose audit")
    print("=" * 82)

    for name, run_dir in parse_specs(args.run):
        r = audit_run(name, run_dir)
        report[name] = r
        print(f"\n{name}: {r['matched_frames']} matched frames")
        print(
            "  online raw center error median/p95: "
            f"{r['online_camera_center_error_raw_m']['median']:.6g} / "
            f"{r['online_camera_center_error_raw_m']['p95']:.6g} m"
        )
        print(
            "  final  raw center error median/p95: "
            f"{r['final_camera_center_error_raw_m']['median']:.6g} / "
            f"{r['final_camera_center_error_raw_m']['p95']:.6g} m"
        )
        print(
            "  online->final center correction median/p95: "
            f"{r['online_to_final_camera_center_correction_m']['median']:.6g} / "
            f"{r['online_to_final_camera_center_correction_m']['p95']:.6g} m"
        )
        print(
            "  online raw rotation error median/p95: "
            f"{r['online_rotation_error_raw_rad']['median']:.6g} / "
            f"{r['online_rotation_error_raw_rad']['p95']:.6g} rad"
        )
        print(
            "  final  raw rotation error median/p95: "
            f"{r['final_rotation_error_raw_rad']['median']:.6g} / "
            f"{r['final_rotation_error_raw_rad']['p95']:.6g} rad"
        )
        print(
            "  aligned translation RMSE online/final: "
            f"{r['online_aligned_translation_rmse_m']:.6g} / "
            f"{r['final_aligned_translation_rmse_m']:.6g} m"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
