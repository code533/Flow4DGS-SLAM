# M2 statistical-object audit

This note fixes the statistical meaning of the uncertainty objects used by M1/M2.
It is an evaluation/provenance document only; it does not change SLAM runtime.

## Pose convention

The repository stores camera poses as world-to-camera transforms (T_{cw}).
M2 absolute uncertainty uses a right perturbation

[
T_{cw} = \bar T_{cw}\operatorname{Exp}(\delta\xi),\qquad
\delta\xi=[\rho,\phi].
]

For a mean pose and a ground-truth pose in the same fixed world gauge, the
corresponding right error is

[
e = \operatorname{Log}(\bar T_{cw}^{-1}T_{cw}^{gt}).
]

Frame 0 is initialized from GT in the current baseline, so the online
trajectory has a fixed initialization gauge. Benchmark ATE is a different
quantity: Flow4DGS rigidly aligns camera-center trajectories before reporting
translation error. That aligned ATE must not be substituted directly for the
right-tangent error used by a 6-DoF covariance.

## Object/target table

| Object | Code object | Statistical meaning | Appropriate validation target |
|---|---|---|---|
| M1 relative parameter covariance | `P_xi_cluster` | Sampling covariance of the fitted flow/depth twist parameter `xi`, conditional on the current residual model/mask | Error of the *same xi parameterization* against the matching relative-motion GT convention. The convention itself must be audited separately. |
| M2-A motion prior | `P_abs_prior_right` | Propagated right-tangent uncertainty of the current motion prior before RGB-D tracking | `Log(T_motion_prior^-1 T_gt)` in the fixed initialization gauge |
| M2-A2 tracking covariance | `P_track_right_raw` | Local sandwich covariance of the RGB-D tracking estimator, conditional on the current Gaussian map, frozen masks/scales, and local residual model | No direct per-frame GT NEES target is currently available. Absolute `Log(T_final^-1 T_gt)` contains map/history bias and is not the same estimand. |
| M2-A2 posterior | `P_abs_post_right` | Information-form combination of the motion-prior covariance and local tracking covariance under a zero-cross-covariance approximation | Absolute final-pose GT error can be shown only as an exploratory end-to-end diagnostic until map uncertainty and prior/tracker dependence are modeled or empirically validated. |
| Benchmark trajectory quality | final ATE | Rigid-aligned camera-center translation error | Report as SLAM accuracy; it is not a 6-DoF covariance-calibration target |

## Consequences for existing experiments

1. The old M2-A2 calibration script fits `P_track_right_raw` to
   `Log(T_final^-1 T_gt)`. That is a useful historical stress test of whether
   a local tracking covariance can proxy total absolute pose error, but it
   must not be described as calibration of the conditional tracking
   covariance itself.
2. The very large Bonn "tracking NEES" therefore does not establish a
   catastrophic tracking failure. The new Bonn runs have final rigid-aligned
   ATE around 2--3 cm.
3. The prior covariance must be evaluated at its own mean
   `T_motion_prior`, not at the post-tracking mean.
4. The prior/tracker innovation audit remains useful as a consistency
   diagnostic, but its zero-cross-covariance assumption is approximate because
   both estimates share history/map information.
5. The GT-fitted SE(3) trajectory-alignment audit is not a valid way to create
   a 6-DoF NEES target. Position-only alignment can make camera centers agree
   while producing a pose transform that is not the tangent-space gauge
   transformation assumed by the covariance calculation.

## Immediate protocol

Do not rerun SLAM for this audit. Existing `m2a_pose_uncertainty/*.pt` files
already contain the means/covariances needed to evaluate the M2-A prior
correctly and to label M2-A2 quantities by their actual estimand.

Until a conditional validation experiment for `P_track_right_raw` is added:

- keep `m2a2_use_diag_calibration: false` for new uncertainty collection;
- do not use the old M2-A2 Diag-6 calibration as a paper result;
- do not tune M2-A3 reliability gates from the old absolute-GT tracking NEES;
- report final ATE separately from covariance consistency.

A later validation of `P_track_right_raw` should use repeated/resampled
tracking observations at a fixed map state (for example a spatial block
bootstrap or controlled observation perturbations), so that empirical
variation of the tracking estimator matches the conditional object estimated
by the sandwich covariance.
