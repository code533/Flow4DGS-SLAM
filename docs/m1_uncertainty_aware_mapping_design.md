# M1 uncertainty-aware Gaussian mapping: minimal method design

Status: design freeze before runtime intervention.

## 1. Evidence boundary

The current evidence supports the local M1 relative-motion uncertainty object,
not a recursively propagated absolute-pose reliability variable.

On the same-domain Bonn box1/box2/box3 LOSO experiment, calibrated M1
uncertainty retains useful held-out ranking information (translation and
rotation Spearman correlations are both about 0.42 in macro average), while
Diag-6 held-out mean NEES is on the same order as the six-dimensional Gaussian
reference. This is sufficient motivation to test M1 as a *local observation
reliability signal*. It is not evidence that M1/M2 predicts total SLAM ATE.

The M2-A/M2-A2/M2-A3 absolute-pose/failure-aware branch is therefore not used
as the main runtime signal in this experiment.

## 2. Statistical object

For frame k, M1 estimates a relative-motion parameter

    xi_k,  P_xi,k

from static flow/depth correspondences. P_xi,k is the cluster/sandwich
covariance in the same local parameterization. A fixed calibration learned on
training sequences may be applied to P_xi,k before it is converted to a scalar
mapping reliability signal.

The mapping method must not interpret this quantity as an absolute camera-pose
covariance.

## 3. Minimal intervention

The first experiment changes only the relative contribution of keyframes to
the RGB-D Gaussian mapping objective. It does NOT:

- change tracking or the camera pose;
- change keyframe selection;
- change Gaussian insertion/deletion;
- change flow fitting;
- change backend pose optimization;
- use M2-A/M2-A2 posterior covariance.

For a keyframe j define

    sigma_t,j = sqrt(trace(P_tt,j) / 3)
    sigma_r,j = sqrt(trace(P_rr,j) / 3).

To avoid mixing metres and radians directly, normalize both components by
training-only reference scales m_t and m_r:

    u_j^2 = 0.5 * [
        (sigma_t,j / m_t)^2 +
        (sigma_r,j / m_r)^2
    ].

Here m_t and m_r are medians of calibrated sigma_t and sigma_r on the
calibration/training sequences. u_j is dimensionless. Larger u_j means lower
local motion confidence.

Convert it to an inverse-uncertainty confidence

    c_j = 1 / (u_j^2 + eps).

Within each mapping window W, normalize confidence to preserve the average loss
scale:

    w_j = c_j / mean_{i in W}(c_i).

The first runtime version should clip w_j only for numerical stability. The
clip bounds are method hyperparameters and must be ablated; they are not
statistical confidence thresholds.

The RGB-D mapping objective becomes

    L_map = sum_{j in W} w_j L_rgbd,j + L_regularization.

Window normalization is important: it makes the experiment primarily a
*relative keyframe reweighting* test instead of silently changing the global
learning-rate/loss scale.

## 4. Why frame/keyframe-level weighting first

M1 is currently a frame-level 6-DoF relative-motion uncertainty. A pixel-wise
mapping weight would require a justified projection of pose uncertainty through
the rendering/measurement Jacobian. That is a separate method and should not be
introduced before the simpler frame-level hypothesis is tested.

Likewise, uncertainty-aware Gaussian insertion is deferred because insertion is
a discrete intervention and introduces an additional threshold. Loss
reweighting gives a cleaner first causal test.

## 5. Deployment/calibration rule

No held-out GT may be used to construct the runtime weight for that sequence.

For LOSO experiments:
- train/calibrate on the training sequences only;
- compute m_t and m_r on the same training sequences;
- freeze those values;
- run the held-out sequence with the frozen calibration/reference scales.

A fold-specific calibration is acceptable for cross-validation because each
fold is fitted without its held-out sequence. A single final model for a paper
benchmark must use a predefined training/calibration split.

## 6. Required pre-runtime audit

Before modifying mapping behavior, verify that the proposed scalar u_j retains
the ranking information seen separately in sigma_t and sigma_r.

For every held-out sequence report:
- Spearman(u, relative translation error);
- Spearman(u, relative rotation error);
- error statistics in low/middle/high uncertainty quantiles;
- dynamic range of u and the implied normalized mapping weights;
- fraction of weights that would hit candidate clip bounds.

Proceed to runtime mapping only if the scalar has a consistent positive
association with held-out local relative-motion error and the weights are not
degenerate.

## 7. First runtime ablation

Keep every baseline setting identical and compare:

1. Flow4DGS baseline / uncertainty OFF.
2. M1 shadow estimation ON, mapping weighting OFF.
3. M1 uncertainty-aware RGB-D mapping ON.

Primary end-task metrics:
- ATE RMSE;
- rendering metrics already supported by the repository (PSNR/SSIM/LPIPS where
  the evaluation protocol provides them);
- runtime overhead.

Uncertainty diagnostics:
- held-out M1 NEES/coverage;
- sigma-error rank correlation;
- distribution of mapping weights.

A useful result must be consistent across multiple sequences. One improved
sequence is not enough to claim an uncertainty-aware mapping benefit.

## 8. Stop rule

If the pre-runtime scalar audit fails, do not implement mapping weighting from
this scalar.

If the scalar audit passes but uncertainty-aware mapping shows no stable
end-task benefit across the predefined evaluation set, stop this intervention
rather than adding keyframe, insertion, pruning, and fusion heuristics at the
same time.
