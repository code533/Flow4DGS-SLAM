#!/usr/bin/env python3
"""Install/check the minimal frontend hook for spatial cross-fitted M1 audit.

Requires the earlier M1 motion-shadow hook to be installed first. The edit is
idempotent and diagnostic-only. Inspect the frontend diff after installation.
"""

import argparse
from pathlib import Path
import sys

FRONTEND = Path("utils/slam_frontend.py")

IMPORT_OLD = "from utils.m1_motion_uncertainty import M1MotionShadowAudit\n"
IMPORT_NEW = IMPORT_OLD + "from utils.m1_motion_crossfit import M1MotionCrossfitAudit\n"

INIT_ANCHOR = "        # M2-A: shadow-only propagation of absolute camera-pose uncertainty.\n"
INIT_BLOCK = '''        # Spatial cross-fitted M1 motion audit. Diagnostic only: four image
        # quadrants are held out in turn and never used in their own twist fit.
        self.m1_motion_crossfit_audit = bool(
            unc_cfg.get("m1_motion_crossfit_audit", False)
        )
        self.m1_motion_crossfit = None
        self.m1_motion_crossfit_frame_range = unc_cfg.get(
            "m1_motion_crossfit_frame_range", None
        )
        if self.m1_motion_crossfit_audit:
            if not self.m1_uncertainty:
                raise ValueError("m1_motion_crossfit_audit requires enable_m1=true")
            signal_file = unc_cfg.get("m1_motion_signal_file")
            signal_fold = unc_cfg.get("m1_motion_signal_fold")
            if not signal_file or not signal_fold:
                raise ValueError(
                    "m1_motion_crossfit_audit requires m1_motion_signal_file and "
                    "m1_motion_signal_fold"
                )
            self.m1_motion_crossfit = M1MotionCrossfitAudit(
                signal_file=signal_file,
                fold=signal_fold,
                eig_floor_rel=float(
                    unc_cfg.get("m1_motion_eig_floor_rel", 1.0e-10)
                ),
            )
            if self.m1_motion_crossfit.block_size not in self.m1_cluster_block_sizes:
                raise ValueError(
                    "M1 motion crossfit requires cluster block size "
                    f"{self.m1_motion_crossfit.block_size}, but configured sizes are "
                    f"{self.m1_cluster_block_sizes}"
                )
            if self.m1_motion_crossfit_frame_range is not None:
                if len(self.m1_motion_crossfit_frame_range) != 2:
                    raise ValueError("m1_motion_crossfit_frame_range must be [lo, hi]")
                self.m1_motion_crossfit_frame_range = [
                    int(v) for v in self.m1_motion_crossfit_frame_range
                ]

'''

CALL_ANCHOR = '''                            dystart=self.config["Training"].get("dystart"),
                        )
                else:
                    xi = xi_baseline
'''
CALL_REPLACEMENT = '''                            dystart=self.config["Training"].get("dystart"),
                        )

                    if self.m1_motion_crossfit_audit:
                        run_crossfit = True
                        if self.m1_motion_crossfit_frame_range is not None:
                            lo, hi = self.m1_motion_crossfit_frame_range
                            run_crossfit = lo <= int(viewpoint.uid) <= hi
                        if run_crossfit:
                            self.m1_motion_crossfit.evaluate_and_save(
                                frame=viewpoint.uid,
                                depth=depth_ds,
                                flow_px=flow_px_ds,
                                K=Kds,
                                flow_var_px2=flow_var_px2,
                                depth_var_m2=depth_var_m2,
                                fb_valid=fb_valid,
                                initial_static_mask=static_prior,
                                baseline_dynamic_mask=results["mask_bool"],
                                static_fit_mask=static_prob_mask,
                                fit_fn=fit_twist_probabilistic,
                                fit_kwargs={
                                    "robust": True,
                                    "iters": 10,
                                    "cauchy_c": self.m1_cauchy_c,
                                    "damping": self.m1_damping,
                                    "residual_rescale": self.m1_residual_rescale,
                                    "min_pixels": self.m1_min_pixels,
                                    "cluster_covariance": True,
                                    "cluster_block_size": self.m1_motion_crossfit.block_size,
                                    "cluster_block_sizes": [self.m1_motion_crossfit.block_size],
                                    "cluster_small_sample_correction": self.m1_cluster_small_sample,
                                },
                                save_dir=self.config["Results"]["save_dir"],
                                dystart=self.config["Training"].get("dystart"),
                            )
                else:
                    xi = xi_baseline
'''


def status(text):
    return {
        "prior_shadow_hook": "self.m1_motion_shadow.evaluate_and_save(" in text,
        "import": "from utils.m1_motion_crossfit import M1MotionCrossfitAudit" in text,
        "init": "self.m1_motion_crossfit_audit = bool(" in text,
        "call": "self.m1_motion_crossfit.evaluate_and_save(" in text,
    }


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"Expected exactly one {label} anchor, found {n}; refusing edit")
    return text.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not FRONTEND.is_file():
        raise FileNotFoundError(FRONTEND)
    text = FRONTEND.read_text()
    st = status(text)
    if args.check:
        print("M1 motion crossfit frontend hook:", st)
        sys.exit(0 if all(st.values()) else 1)
    if not st["prior_shadow_hook"]:
        raise RuntimeError(
            "Install scripts/install_m1_motion_shadow.py first, then rerun this installer"
        )
    if st["import"] and st["init"] and st["call"]:
        print("M1 motion crossfit frontend hook already installed; no changes made.")
        return
    if st["import"] or st["init"] or st["call"]:
        raise RuntimeError(f"Partial crossfit hook detected {st}; refusing ambiguous edit")
    text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import")
    text = replace_once(text, INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR, "init")
    text = replace_once(text, CALL_ANCHOR, CALL_REPLACEMENT, "runtime call")
    FRONTEND.write_text(text)
    print("Installed M1 spatial crossfit shadow hook.")
    print("Next: git diff --check && python -m py_compile utils/slam_frontend.py utils/m1_motion_crossfit.py")


if __name__ == "__main__":
    main()
