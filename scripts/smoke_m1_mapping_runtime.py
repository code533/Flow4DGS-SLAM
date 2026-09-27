#!/usr/bin/env python3
"""Static smoke checks for the M1 uncertainty-aware mapping experiment.

No CUDA or dataset is required. This catches configuration/report mismatches
before a costly SLAM run.
"""

import argparse
import ast
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.config_utils import load_config
from utils.m1_mapping_uncertainty import M1MappingSignal, normalized_window_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more *_m1map.yaml experiment configs.",
    )
    args = parser.parse_args()

    for rel in args.config:
        path = Path(rel)
        cfg = load_config(str(path))
        unc = cfg.get("Uncertainty", {})
        assert unc.get("enable_m1", False), f"{path}: enable_m1 must be true"
        assert unc.get("m1_mapping_weighting", False), (
            f"{path}: m1_mapping_weighting must be true"
        )
        signal = M1MappingSignal(
            unc["m1_mapping_signal_file"],
            unc["m1_mapping_signal_fold"],
            eig_floor_rel=float(unc.get("m1_mapping_eig_floor_rel", 1e-10)),
            eps=float(unc.get("m1_mapping_eps", 1e-6)),
        )
        assert signal.block_size in [
            int(v) for v in unc.get("cluster_block_sizes", [])
        ]
        w = normalized_window_weights(
            [0.2, 0.5, 1.0, 2.0],
            clip_min=float(unc.get("m1_mapping_clip_min", 0.25)),
            clip_max=float(unc.get("m1_mapping_clip_max", 4.0)),
        )
        assert abs(sum(w) / len(w) - 1.0) < 1e-12
        print(
            f"OK {path}: fold={signal.fold}, mode={signal.mode}, "
            f"block={signal.block_size}, refs=({signal.ref_t:.6g}, "
            f"{signal.ref_r:.6g}), test_weights={w}"
        )

    for rel in [
        "utils/m1_mapping_uncertainty.py",
        "utils/slam_frontend.py",
        "utils/slam_backend.py",
        "utils/camera_utils.py",
    ]:
        source = (REPO / rel).read_text()
        ast.parse(source, filename=rel)
        print("syntax OK", rel)


if __name__ == "__main__":
    main()
