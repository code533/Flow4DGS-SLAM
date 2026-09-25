# M2: Pose uncertainty to dynamic 3D observation uncertainty

## Goal

M2 converts the camera-motion uncertainty estimated in M1 into uncertainty of
3D RGB-D observations used by the dynamic Gaussian representation.

The first M2 stage is deliberately shadow-only: derive and audit uncertainty
propagation without changing Flow4DGS tracking, mapping, deformation, keyframe
selection, densification, or pruning.

## Repository insertion points

The current code uses world-to-camera poses in Camera.w2c. Dynamic Gaussians
are created from keyframe RGB-D data through Backend.add_next_node() ->
GaussianModel.extend_from_pcd_seq(). Dynamic deformation is optimized in
Backend.map(), including bidirectional optical-flow supervision between a
viewpoint and its closest keyframe.

M2 therefore has two downstream uses:

1. per-observation world-space 3D covariance during dynamic Gaussian
   initialization/update;
2. later, uncertainty-aware weighting of dynamic flow/deformation factors.

This stage implements only (1) plus convention audits.

## Pose covariance convention

Absolute camera uncertainty is represented as a right-invariant perturbation

    T_cw = Tbar_cw Exp(delta_xi^),
    delta_xi ~ N(0, P_cw^R),

where T_cw is world-to-camera and delta_xi = [rho, phi].

For the Flow4DGS composition

    T_k = T_(k-1) DeltaT_k,

right-invariant covariance propagates as

    P_k^R = A P_(k-1)^R A^T + P_Delta^R,
    A = Ad_(DeltaT_k^-1),

under the first-order independence assumption.

M1 stores covariance in the exponential-coordinate parameter xi of

    DeltaT = Exp(xi).

Before propagation, it is converted to a right group-error covariance using

    epsilon = Log(DeltaT^-1 Exp(xi + dxi)),
    P_Delta^R = J_r P_xi J_r^T.

The initial implementation evaluates J_r numerically by a six-dimensional
central finite difference to avoid assuming J_r = I.

## RGB-D point uncertainty

For pixel u = [u,v], depth d, and q = K^-1 [u,v,1]^T,

    x_c = d q,
    X_w = T_cw^-1 x_c.

With right-invariant pose perturbation,

    J_pose = dX_w / d(delta_xi)
           = [-I, [X_w]_x].

Depth propagation is

    J_d = R_cw^T q.

Optional pixel-coordinate propagation is

    J_uv = R_cw^T
           [[d/fx, 0],
            [0, d/fy],
            [0, 0]].

The 3D covariance is

    Sigma_X =
        J_pose P_cw^R J_pose^T
        + J_d sigma_d^2 J_d^T
        + J_uv R_uv J_uv^T.

A compact scalar uncertainty for diagnostics is

    sigma_X,rms = sqrt(trace(Sigma_X) / 3).

## Important bridge from M1 to M2

M1 uncertainty is estimated before the subsequent photometric/depth tracking
iterations. Flow4DGS then updates the pose mean with left-multiplicative
increments:

    T <- Exp(tau) T.

A deterministic left re-centering does not change right-invariant error
coordinates, which is one reason M2 uses right-invariant absolute covariance.
However, tracking also adds information and could reduce the true posterior
uncertainty. The initial M2 implementation therefore treats M1 covariance as a
motion-prior uncertainty and will explicitly log the pre/post tracking pose
correction before deciding whether a tracking-information update is required.

## Current implementation

- utils/m2_uncertainty.py
  - SE(3) adjoint
  - M1 xi covariance -> right group-error covariance
  - recursive right-invariant pose covariance propagation
  - pose/depth/pixel -> world-point covariance
  - RMS 3D uncertainty

- scripts/audit_m2_uncertainty_jacobians.py
  - analytic-vs-finite-difference point Jacobian check
  - adjoint convention check
  - covariance symmetry/PSD check

Run:

    python scripts/audit_m2_uncertainty_jacobians.py

Only after this audit passes should M2 be wired into live Camera objects and
dynamic Gaussian observations.


## M2-A live shadow propagation

M2-A is now wired into the frontend in shadow mode.

Configuration:

```yaml
Uncertainty:
  enable_m1: true
  enable_m2a: true
  m2a_save_diagnostics: true
  m2a_use_diag_calibration: false
  m2a_diag_calibration: null
  m2a_right_jacobian_eps: 1.0e-5
```

At frame 0, the covariance is anchored to zero because the baseline explicitly
initializes that frame with the ground-truth pose.

For each subsequent frame:

1. take the M1 32x32 cluster covariance in xi-parameter coordinates;
2. optionally apply the externally frozen Diag-6 calibration in that same
   parameter space;
3. convert to a right-invariant relative group covariance;
4. propagate the previous absolute covariance using the relative pose actually
   applied after the Flow4DGS motion cap;
5. attach both raw and selected absolute covariance to the Camera object;
6. keep covariance unchanged through the later photometric tracking iterations
   and log the deterministic pose correction magnitude.

This last choice is intentionally an audit approximation. The photometric
tracking stage adds information, so a future posterior update may reduce the
pose covariance. M2-A first measures how large that correction is before
introducing another information model.

Diagnostics are written to:

```
<Results.save_dir>/m2a_pose_uncertainty/*.pt
```

Analyze them with:

```bash
python scripts/analyze_m2a_pose_uncertainty.py \
  results/m2a_pose_uncertainty
```

The analysis reports numerical PSD/symmetry checks, absolute-pose NEES and
coverage against GT, uncertainty-error correlation, and the magnitude of the
post-prior tracking correction.
