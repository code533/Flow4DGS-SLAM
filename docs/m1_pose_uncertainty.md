# M1: Probabilistic Camera-Motion Estimation

This branch implements the first milestone of the uncertainty-propagation
project while preserving the original Flow4DGS-SLAM motion mask and mapping.

## Scope

M1 upgrades the second static-pixel camera-motion refit from

```
xi = fit_twist_weighted(...)
```

to a generalized least-squares estimator that also returns a local relative
pose covariance:

```
(xi, P_delta_T) = fit_twist_probabilistic(...)
```

No Gaussian-state uncertainty is propagated in M1.

## Measurement model

For a candidate static pixel p,

```
f_p = L_p xi + eps_p
```

with residual covariance

```
R_p = sigma_F,p^2 I_2 + J_D,p sigma_D,p^2 J_D,p^T.
```

The optical-flow variance proxy is derived from forward/backward RAFT
consistency. The initial depth model is intentionally simple:

```
sigma_D = depth_sigma0_m + depth_sigma_rel * abs(depth).
```

Both models are calibration targets rather than final statistical claims.

## Local covariance approximation

The final robust information matrix is

```
H = sum_p L_p^T W_p L_p
```

where the Cauchy weight is applied to the 2-D Mahalanobis innovation. M1
reports

```
P_delta_T ~= kappa * pinv(H).
```

This is a local Gauss-Newton/Laplace covariance approximation conditioned on
the final robust weights, not an exact Bayesian posterior.

## Baseline isolation

M1 deliberately keeps the original dynamic mask unchanged. The probabilistic
solver replaces only the second static-pixel pose refit. This isolates the
effect of heteroscedastic flow/depth weighting and makes baseline comparisons
interpretable.

The original Flow4DGS-SLAM motion cap remains enabled after the refit. M1 logs
both the raw relative motion and the actually applied relative transform.

## Configuration

The dataset base configs contain a disabled-by-default section:

```yaml
Uncertainty:
  enable_m1: false
  flow_sigma0_px: 0.5
  flow_fb_lambda: 1.0
  flow_max_sigma_px: 20.0
  depth_sigma0_m: 0.01
  depth_sigma_rel: 0.0
  cauchy_c: 2.0
  damping: 1.0e-6
  min_pixels: 500
  residual_rescale: false
  save_diagnostics: true
```

Set `enable_m1: true` in the effective config to activate M1.

## Diagnostics

When enabled, per-frame files are written to:

```
<save_dir>/m1_pose_uncertainty/000001.pt
...
```

Each file contains:

- `xi_raw`: raw GLS twist estimate.
- `P_xi_raw`: 6x6 local covariance approximation.
- `T_rel_raw`: raw SE(3) relative transform.
- `T_rel_applied`: relative transform after the original Flow4DGS motion cap.
- `sigma_trans_m`: standard deviations of the three translation components.
- `sigma_rot_rad`: standard deviations of the three rotation components.
- `kappa`: diagnostic residual scale. It is logged but, by default, no longer shrinks the covariance.
- `condition`: condition number of the final information matrix.
- `num_pixels`: number of pixels used by the probabilistic refit.
- `fb_error_median_px`: median forward/backward flow consistency error.
- `maha_median`: median normalized 2-D innovation.
- motion-cap scale diagnostics.

## M1 validation checklist

Before propagating pose uncertainty into the dynamic Gaussian state:

1. Verify every `P_xi_raw` is finite, symmetric and positive semidefinite up
   to numerical tolerance.
2. Check that difficult frames exhibit larger covariance and/or worse
   information-matrix conditioning.
3. Compare relative-pose error against predicted covariance.
4. Evaluate translation and rotation uncertainty separately; do not combine
   meter and radian variances into a single trace.
5. Once ground-truth relative poses are available, evaluate NEES:

```
NEES = delta_xi^T P_xi^{-1} delta_xi.
```

6. Calibrate the forward/backward consistency-to-variance mapping before
   interpreting chi-square thresholds probabilistically.

## Current approximations

- Flow covariance is isotropic per pixel.
- Flow/depth errors are treated as independent.
- Depth uncertainty uses a simple analytic noise model.
- Robust weights are treated as fixed when computing the local covariance.
- M1 does not yet propagate absolute pose covariance.
- M1 does not change the motion mask, mapping, Gaussian states, or rendering.


## GT calibration pass

Current M1 diagnostics also save `T_rel_gt`, constructed in the same
right-composed convention as the raw Flow4DGS relative update. The analysis
script therefore reports:

- raw relative translation and rotation error,
- applied-motion error after the original Flow4DGS motion cap,
- Pearson and Spearman error/uncertainty correlation,
- 6-DoF, translation-only, and rotation-only NEES,
- chi-square coverage,
- uncertainty-quantile calibration tables.

By default `residual_rescale: false`. The previously logged `kappa` often
fell below one because robust Cauchy weighting suppresses the normalized
residuals; using it as a covariance multiplier would make the already-local
Gauss-Newton covariance more overconfident. We retain `kappa` as a
diagnostic until calibration supports a principled rescaling rule.

Existing diagnostic files created before `T_rel_gt` was added can still be
analyzed numerically, but GT correlation and NEES require a new M1 run.


## M1.1: spatial correlation-aware covariance

Dense RAFT flow pixels are strongly spatially correlated. Treating tens of
thousands of pixels as independent makes the naive inverse-Hessian covariance
severely over-confident. M1.1 therefore reports both:

```
P_H = H^{-1}
```

and a spatial cluster-robust sandwich covariance:

```
P_CR = A^{-1} B A^{-1},
B = sum_b S_b S_b^T,
S_b = sum_{p in block b} J_p^T W_p r_p.
```

Pixels inside one image block may be arbitrarily correlated; approximate
independence is only assumed across blocks. The default block size is 32x32.
The pose mean is unchanged by this covariance upgrade.

Configuration:

```yaml
Uncertainty:
  cluster_covariance: true
  cluster_block_size: 32
  cluster_small_sample_correction: true
```

New diagnostics save `P_xi_hessian`, `P_xi_cluster`, the selected
`P_xi_raw`, cluster count, block size, and covariance mode.

The analyzer now also performs a pose-convention audit. Since the fitted
covariance is the covariance of the twist parameter `xi`, its primary NEES
uses

```
delta_xi = xi_est - Log(T_rel_gt)
```

rather than only the group-space error `Log(T_rel_gt^{-1} T_rel_est)`.
It also tests the inverse GT convention as a sanity check.


## M1.2: strict shadow mode and cache isolation

M1 now defaults to strict shadow mode:

```yaml
Uncertainty:
  shadow_mode: true
  log_keyframe_reasons: true
```

In shadow mode, the original Flow4DGS pose mean from
`fit_twist_weighted(..., iters=30)` remains the pose used by tracking,
mapping, and keyframe selection. The probabilistic module only evaluates
covariance around that fixed baseline twist. This prevents uncertainty
experiments from silently changing the SLAM state trajectory.

The bidirectional flow used for forward/backward consistency is also computed
with `cache=False`. This prevents the temporary current<->previous flow pair
from populating `Camera.flow` / `Camera.flow_back`, which are later reused
by backend mapping against a potentially different keyframe.

Keyframe creation now optionally logs the trigger:

- frame index,
- previous keyframe index,
- frame gap,
- geometric decision,
- forced gap-of-5 trigger,
- dynamic-start trigger,
- new-object trigger.

With shadow mode enabled, M1 should not intentionally change the baseline
pose mean or keyframe-selection inputs. Remaining keyframe differences should
therefore be investigated as scheduling/non-determinism or another unrelated
state mutation rather than as an uncertainty-estimator effect.


## Multi-scale block-size ablation

M1 now computes several cluster-robust covariance variants from the same final
pixel-score field in a single SLAM run. By default:

```yaml
Uncertainty:
  cluster_block_size: 32
  cluster_block_sizes: [16, 32, 64]
```

`cluster_block_size` selects the covariance used as the main M1 output,
while `cluster_block_sizes` controls the ablation set saved for analysis.

Each diagnostic file now contains:

- `P_xi_clusters[16]`
- `P_xi_clusters[32]`
- `P_xi_clusters[64]`
- per-scale cluster counts

The analysis script prints Pearson/Spearman correlation, NEES, coverage, and
uncertainty quintiles for every requested block size. This avoids rerunning
the complete SLAM pipeline only to change the cluster partition.


## Cross-sequence calibration

Use `scripts/calibrate_m1_multiseq.py` to fit a sequence-independent
translation/rotation scale on calibration sequences and evaluate it on held-out
sequences.

The current calibration model is intentionally simple:

```
P_cal = S P_cluster S^T
S = diag(s_t, s_t, s_t, s_r, s_r, s_r)
```

`s_t` and `s_r` are fitted only from the training split by matching the
mean translation- and rotation-marginal NEES to their 3-DoF expected value.

Example:

```bash
python scripts/calibrate_m1_multiseq.py \
  --block-size 32 \
  --train walking_xyz=/path/run1/m1_pose_uncertainty \
          sitting_static=/path/run2/m1_pose_uncertainty \
  --test bonn_placing=/path/run3/m1_pose_uncertainty \
         sitting_rpy=/path/run4/m1_pose_uncertainty \
  --output results/m1_calibration_block32.json
```

The report includes raw and calibrated pooled/per-sequence NEES, chi-square
coverage, and Pearson/Spearman ranking metrics. If held-out NEES remains
strongly sequence-dependent after this scale calibration, that is evidence for
adding an explicit model-discrepancy term `Q_model` rather than further
tuning the cluster block size.


## Full whitened 6x6 calibration

The cross-sequence calibration script now compares three models:

```
P_raw   = P_cluster
P_scale = S P_cluster S^T
P_full  = P_cluster^(1/2) C P_cluster^(1/2)
```

The full calibration is fitted on training sequences only:

```
z_i = P_i^(-1/2) e_i
C   = mean_i z_i z_i^T
```

It can therefore correct anisotropic DoF scaling and translation-rotation
coupling that the two-scalar baseline cannot represent.

Example:

```bash
python scripts/calibrate_m1_multiseq.py \
  --block-size 32 \
  --train walking_static=/path/run1/m1_pose_uncertainty \
          sitting_rpy=/path/run2/m1_pose_uncertainty \
  --test bonn_placing=/path/run3/m1_pose_uncertainty \
         sitting_static=/path/run4/m1_pose_uncertainty \
  --output results/m1_calibration_block32_full.json
```

The script prints the learned dimensionless 6x6 matrix `C`, its eigenvalues,
condition number, translation-rotation cross-block norm, and held-out metrics
for raw, two-scale, and full-6x6 calibration.

Optional regularization:

```
--full-shrinkage 0.1
```

shrinks `C` toward an isotropic matrix with the same trace. The default is
zero shrinkage so that the first experiment directly tests whether covariance
shape/cross-correlation calibration explains the remaining NEES gap.


## Structured whitened calibration ablation

The calibration script now compares three structured variants derived from the
same training-set whitened second-moment matrix `C`:

```
Diag-6:
  C_diag = diag(diag(C))

Block-6:
  C_block = blockdiag(C_tt, C_rr)

Full-6:
  C_full = C
```

The corresponding calibrated covariance is always

```
P_cal = P_cluster^(1/2) C_* P_cluster^(1/2).
```

This ablation isolates whether held-out improvements are explained by:

1. per-DoF anisotropic scale only,
2. within-translation / within-rotation coupling,
3. translation-rotation cross coupling.

No SLAM rerun is required. Re-run only
`scripts/calibrate_m1_multiseq.py` on the existing M1 diagnostic folders.
