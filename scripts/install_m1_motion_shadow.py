#!/usr/bin/env python3
"""Install/check the minimal frontend hook for M1 motion shadow diagnostics.

The connector used to prepare this experiment can add new repository files but
cannot safely apply a partial edit to the large slam_frontend.py. This script
therefore performs three exact, idempotent source edits locally. Always inspect
`git diff -- utils/slam_frontend.py` after applying it.
"""

import argparse
from pathlib import Path
import sys


FRONTEND = Path("utils/slam_frontend.py")

IMPORT_OLD = "from utils.m1_mapping_uncertainty import M1MappingSignal\n"
IMPORT_NEW = (
    IMPORT_OLD
    + "from utils.m1_motion_uncertainty import M1MotionShadowAudit\n"
)

INIT_ANCHOR = """        # M2-A: shadow-only propagation of absolute camera-pose uncertainty.\n"""
INIT_BLOCK = """        # Diagnostic-only propagation of M1 pose covariance into rigid-flow\n        # residuals. This path writes summaries but never changes the motion mask.\n        self.m1_motion_shadow_audit = bool(\n            unc_cfg.get(\"m1_motion_shadow_audit\", False)\n        )\n        self.m1_motion_shadow = None\n        if self.m1_motion_shadow_audit:\n            if not self.m1_uncertainty:\n                raise ValueError(\"m1_motion_shadow_audit requires enable_m1=true\")\n            signal_file = unc_cfg.get(\"m1_motion_signal_file\")\n            signal_fold = unc_cfg.get(\"m1_motion_signal_fold\")\n            if not signal_file or not signal_fold:\n                raise ValueError(\n                    \"m1_motion_shadow_audit requires m1_motion_signal_file and \"\n                    \"m1_motion_signal_fold\"\n                )\n            self.m1_motion_shadow = M1MotionShadowAudit(\n                signal_file=signal_file,\n                fold=signal_fold,\n                eig_floor_rel=float(\n                    unc_cfg.get(\"m1_motion_eig_floor_rel\", 1.0e-10)\n                ),\n                hist_bins=int(unc_cfg.get(\"m1_motion_hist_bins\", 64)),\n            )\n            if self.m1_motion_shadow.block_size not in self.m1_cluster_block_sizes:\n                raise ValueError(\n                    \"M1 motion shadow requires cluster block size \"\n                    f\"{self.m1_motion_shadow.block_size}, but configured sizes are \"\n                    f\"{self.m1_cluster_block_sizes}\"\n                )\n\n"""

CALL_OLD = """                        viewpoint.m1_mapping_valid = True\n                else:\n                    xi = xi_baseline\n"""
CALL_NEW = """                        viewpoint.m1_mapping_valid = True\n\n                    if self.m1_motion_shadow_audit:\n                        block_size = self.m1_motion_shadow.block_size\n                        cov_for_motion = m1_result[\"cov_clusters\"].get(block_size)\n                        if cov_for_motion is None:\n                            raise KeyError(\n                                f\"M1 runtime result has no cluster covariance for \"\n                                f\"block size {block_size}\"\n                            )\n                        self.m1_motion_shadow.evaluate_and_save(\n                            frame=viewpoint.uid,\n                            depth=depth_ds,\n                            flow_px=flow_px_ds,\n                            K=Kds,\n                            xi=xi_baseline,\n                            P_raw=cov_for_motion,\n                            flow_var_px2=flow_var_px2,\n                            depth_var_m2=depth_var_m2,\n                            fb_valid=fb_valid,\n                            initial_static_mask=static_prior,\n                            baseline_dynamic_mask=results[\"mask_bool\"],\n                            save_dir=self.config[\"Results\"][\"save_dir\"],\n                            covariance_mode=m1_result[\"covariance_mode\"],\n                            dystart=self.config[\"Training\"].get(\"dystart\"),\n                        )\n                else:\n                    xi = xi_baseline\n"""


def status(text):
    return {
        "import": "from utils.m1_motion_uncertainty import M1MotionShadowAudit" in text,
        "init": "self.m1_motion_shadow_audit = bool(" in text,
        "call": "self.m1_motion_shadow.evaluate_and_save(" in text,
    }


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {label} anchor, found {count}. "
            "Refusing to edit slam_frontend.py."
        )
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only verify hook status")
    args = parser.parse_args()

    if not FRONTEND.is_file():
        raise FileNotFoundError(
            f"{FRONTEND} not found; run this script from the repository root"
        )
    text = FRONTEND.read_text()
    st = status(text)

    if args.check:
        print("M1 motion shadow frontend hook:", st)
        sys.exit(0 if all(st.values()) else 1)

    if all(st.values()):
        print("M1 motion shadow frontend hook is already installed; no changes made.")
        return
    if any(st.values()):
        raise RuntimeError(
            f"Partial hook detected {st}; refusing an ambiguous edit. "
            "Inspect utils/slam_frontend.py manually."
        )

    text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import")
    text = replace_once(text, INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR, "init")
    text = replace_once(text, CALL_OLD, CALL_NEW, "runtime call")
    FRONTEND.write_text(text)
    print("Installed M1 motion shadow frontend hook.")
    print(
        "Next: git diff --check && python -m py_compile "
        "utils/slam_frontend.py utils/m1_motion_uncertainty.py"
    )


if __name__ == "__main__":
    main()
