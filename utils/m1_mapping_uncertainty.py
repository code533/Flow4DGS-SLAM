"""Runtime helper for the audited M1 mapping-reliability signal.

This module reproduces the covariance calibration and dimensionless scalar used
by scripts/audit_m1_mapping_signal.py. It intentionally operates on the local
M1 relative-motion covariance only; it does not interpret the result as an
absolute-pose uncertainty.
"""

import json
from pathlib import Path

import torch


def _sym(P):
    return 0.5 * (P + P.T)


def _sqrt_psd(P, eig_floor_rel=1e-10):
    P = _sym(P.double())
    eig, vec = torch.linalg.eigh(P)
    max_eig = eig.max().clamp_min(1e-18)
    eig = eig.clamp_min(max_eig * float(eig_floor_rel))
    root = (vec * torch.sqrt(eig).unsqueeze(0)) @ vec.T
    return _sym(root)


def _resolve_calibration_path(signal_file, value):
    path = Path(value).expanduser()
    if path.is_file():
        return path
    # Common case: report stores "results/foo.json" and SLAM is launched from
    # another working directory. Try relative to repository root inferred from
    # the signal report's parent when possible.
    candidate = signal_file.parent / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Could not resolve calibration file {value!r} referenced by "
        f"{signal_file}"
    )


class M1MappingSignal:
    def __init__(self, signal_file, fold, eig_floor_rel=1e-10, eps=1e-6):
        self.signal_file = Path(signal_file).expanduser()
        self.fold = str(fold)
        self.eig_floor_rel = float(eig_floor_rel)
        self.eps = float(eps)

        with self.signal_file.open("r") as f:
            signal_report = json.load(f)

        if signal_report.get("method") != "m1_dimensionless_mapping_signal_audit":
            raise ValueError(
                "Expected output from scripts/audit_m1_mapping_signal.py, got "
                f"method={signal_report.get('method')!r}"
            )
        self.block_size = int(signal_report["block_size"])
        self.mode = str(signal_report["calibration_mode"])
        if self.fold not in signal_report.get("folds", {}):
            raise KeyError(
                f"Signal report has no fold {self.fold!r}; available: "
                f"{sorted(signal_report.get('folds', {}).keys())}"
            )

        fold_signal = signal_report["folds"][self.fold]
        refs = fold_signal["reference_scales_training_only"]
        self.ref_t = float(refs["median_sigma_t_m"])
        self.ref_r = float(refs["median_sigma_r_rad"])
        if self.ref_t <= 0 or self.ref_r <= 0:
            raise ValueError("Training-only reference scales must be positive")

        calibration_file = _resolve_calibration_path(
            self.signal_file, signal_report["calibration_file"]
        )
        with calibration_file.open("r") as f:
            calibration_report = json.load(f)

        if calibration_report.get("mode") != "loso":
            raise ValueError("Runtime mapping signal currently requires LOSO calibration")
        fold_cal = calibration_report.get("folds", {}).get(self.fold)
        if fold_cal is None:
            raise KeyError(
                f"Calibration report has no fold {self.fold!r}"
            )

        variants = (
            fold_cal.get("calibration", {})
            .get("whitened", {})
            .get("variants", {})
        )
        if self.mode not in variants:
            raise KeyError(
                f"Calibration fold {self.fold!r} has no variant {self.mode!r}"
            )
        self.C = _sym(
            torch.tensor(variants[self.mode]["C"], dtype=torch.float64)
        )
        if tuple(self.C.shape) != (6, 6):
            raise ValueError("Expected a 6x6 M1 calibration matrix")

    def calibrated_covariance(self, P_raw):
        root = _sqrt_psd(P_raw, eig_floor_rel=self.eig_floor_rel)
        return _sym(root @ self.C @ root)

    def evaluate(self, P_raw):
        P_cal = self.calibrated_covariance(P_raw)
        diag = torch.diagonal(P_cal).clamp_min(0.0)
        sigma_t = float(torch.sqrt(diag[:3].mean()))
        sigma_r = float(torch.sqrt(diag[3:].mean()))
        u2 = 0.5 * (
            (sigma_t / self.ref_t) ** 2
            + (sigma_r / self.ref_r) ** 2
        )
        u = float(u2 ** 0.5)
        confidence = float(1.0 / (u2 + self.eps))
        return {
            "sigma_t": sigma_t,
            "sigma_r": sigma_r,
            "u": u,
            "confidence": confidence,
        }


def normalized_window_weights(
    confidences,
    clip_min=0.25,
    clip_max=4.0,
):
    """Normalize, clip, then renormalize positive confidence values."""
    c = torch.as_tensor(confidences, dtype=torch.float64)
    if c.numel() == 0:
        return []
    if not bool(torch.isfinite(c).all()) or bool((c <= 0).any()):
        raise ValueError("Mapping confidences must be finite and positive")

    w = c / c.mean().clamp_min(1e-18)
    if clip_min is not None or clip_max is not None:
        lo = -float("inf") if clip_min is None else float(clip_min)
        hi = float("inf") if clip_max is None else float(clip_max)
        w = torch.clamp(w, min=lo, max=hi)
        w = w / w.mean().clamp_min(1e-18)
    return [float(v) for v in w]
