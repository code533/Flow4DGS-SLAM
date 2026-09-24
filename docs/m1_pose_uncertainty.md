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
