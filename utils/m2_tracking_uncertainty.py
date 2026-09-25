"""M2-A2 tracking-posterior uncertainty utilities.

This module treats the final RGB-D tracking result as a pose observation that
updates the M1/M2-A motion prior.

The tracking covariance is estimated in the LEFT pose tangent because the
Flow4DGS rasterizer's (rho, theta) camera deltas are applied as

    T' = Exp([rho,theta]) T.

A finite-difference residual Jacobian is combined with a spatial
cluster-robust sandwich estimator. The resulting left-tangent covariance is
then converted to the right-invariant tangent used by M2-A recursion.
"""

import math

import torch

from utils.m2_uncertainty import adjoint_SE3


def _mad_scale(values, floor):
    if values.numel() == 0:
        return torch.as_tensor(
            float(floor), device=values.device, dtype=values.dtype
        )
    med = values.median()
    mad = (values - med).abs().median()
    # Gaussian-consistent MAD scale. Keep a floor for nearly perfect renders.
    return torch.clamp(1.4826 * mad, min=float(floor))


def build_tracking_residual_context(
    config,
    render_pkg,
    viewpoint,
    rm_dynamic=True,
    extra_mask=None,
    rgb_scale_floor=0.01,
    depth_scale_floor=0.01,
):
    """Build fixed masks/scales for M2-A2 finite-difference residuals.

    The masks are frozen at the final tracking mean to avoid differentiating
    through threshold decisions such as opacity > 0.95. Residuals retain the
    baseline exposure and opacity weighting used by Flow4DGS tracking.
    """
    image = render_pkg["render"]
    depth = render_pkg["depth"]
    opacity = render_pkg["opacity"]

    gt_image = viewpoint.original_image.to(
        device=image.device, dtype=image.dtype
    )
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        device=depth.device, dtype=depth.dtype
    )[None]

    H, W = depth.shape[-2:]
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_mask = (
        gt_image.sum(dim=0) > rgb_boundary_threshold
    ).view(1, H, W)

    if viewpoint.grad_mask is not None:
        gm = viewpoint.grad_mask.to(device=image.device).bool()
        if gm.ndim == 2:
            gm = gm[None]
        rgb_mask = rgb_mask & gm

    depth_mask = (
        (gt_depth > 0.01)
        & (gt_depth < 1000.0)
        & (opacity.detach() > 0.95)
    ).view(1, H, W)

    if (
        viewpoint.motion_mask is not None
        and rm_dynamic
        and viewpoint.uid > 0
    ):
        motion = viewpoint.motion_mask.to(device=image.device).bool()
        motion = motion.view(1, H, W)
        rgb_mask = rgb_mask & motion
        depth_mask = depth_mask & motion

    if extra_mask is not None:
        em = extra_mask.to(device=image.device).bool().view(1, H, W)
        rgb_mask = rgb_mask & em
        depth_mask = depth_mask & em

    image_ab = (
        torch.exp(viewpoint.exposure_a.detach()) * image
        + viewpoint.exposure_b.detach()
    )

    # Signed residuals. RGB keeps the baseline opacity multiplier.
    rgb_raw = opacity.detach() * (image_ab - gt_image)
    depth_raw = depth - gt_depth

    rgb_valid = rgb_mask.expand(3, -1, -1)
    depth_valid = depth_mask

    rgb_scale = _mad_scale(
        rgb_raw[rgb_valid].detach(), rgb_scale_floor
    )
    depth_scale = _mad_scale(
        depth_raw[depth_valid].detach(), depth_scale_floor
    )

    alpha = float(config["Training"].get("alpha", 0.95))
    # Relative RGB/depth weighting follows the baseline tracking objective.
    # Division by 3 compensates for the three RGB channels.
    rgb_weight = math.sqrt(max(alpha / 3.0, 1e-12))
    depth_weight = math.sqrt(max(1.0 - alpha, 1e-12))

    return {
        "rgb_mask": rgb_mask.detach(),
        "depth_mask": depth_mask.detach(),
        "gt_image": gt_image.detach(),
        "gt_depth": gt_depth.detach(),
        "rgb_scale": rgb_scale.detach(),
        "depth_scale": depth_scale.detach(),
        "rgb_weight": rgb_weight,
        "depth_weight": depth_weight,
        "H": H,
        "W": W,
    }


def tracking_residual_tensor(render_pkg, viewpoint, context):
    """Return normalized signed residual maps [4,H,W] and validity mask."""
    image = render_pkg["render"]
    depth = render_pkg["depth"]
    opacity = render_pkg["opacity"]

    image_ab = (
        torch.exp(viewpoint.exposure_a.detach()) * image
        + viewpoint.exposure_b.detach()
    )

    rgb = (
        context["rgb_weight"]
        * opacity
        * (image_ab - context["gt_image"])
        / context["rgb_scale"]
    )
    dep = (
        context["depth_weight"]
        * (depth - context["gt_depth"])
        / context["depth_scale"]
    )

    residual = torch.cat([rgb, dep], dim=0)
    valid = torch.cat(
        [
            context["rgb_mask"].expand(3, -1, -1),
            context["depth_mask"],
        ],
        dim=0,
    )
    residual = torch.where(valid, residual, torch.zeros_like(residual))
    return residual, valid


def cluster_robust_tracking_covariance(
    residual,
    jacobian,
    valid,
    block_size=32,
    cauchy_c=2.0,
    damping=1e-6,
    small_sample_correction=True,
):
    """Cluster-robust covariance of a 6D tracking pose observation.

    residual: [C,H,W]
    jacobian: [C,H,W,6]
    valid:    [C,H,W]

    Channels at the same pixel are combined before spatial clustering.
    """
    if residual.ndim != 3 or jacobian.ndim != 4:
        raise ValueError("Unexpected tracking residual/Jacobian shape")
    C, H, W = residual.shape
    if jacobian.shape != (C, H, W, 6):
        raise ValueError("jacobian must have shape [C,H,W,6]")

    r = residual.permute(1, 2, 0).reshape(-1, C)
    J = jacobian.permute(1, 2, 0, 3).reshape(-1, C, 6)
    m = valid.permute(1, 2, 0).reshape(-1, C)

    # Robust Cauchy weights in normalized residual units.
    c2 = float(cauchy_c) ** 2
    w = 1.0 / (1.0 + (r * r) / c2)
    w = w * m.to(dtype=r.dtype)

    active_pixel = m.any(dim=1)
    num_pixels = int(active_pixel.sum())
    num_observations = int(m.sum())

    if num_pixels < 8 or num_observations < 12:
        huge = torch.eye(
            6, device=r.device, dtype=r.dtype
        ) * 1e3
        return {
            "cov": huge,
            "information": torch.zeros_like(huge),
            "cluster_count": 0,
            "num_pixels": num_pixels,
            "num_observations": num_observations,
            "condition": torch.tensor(
                float("inf"), device=r.device, dtype=r.dtype
            ),
        }

    A = torch.einsum("nci,nc,ncj->ij", J, w, J)
    score_pixel = torch.einsum("nci,nc,nc->ni", J, w, r)

    mean_diag = torch.diagonal(A).mean().abs().clamp_min(1e-12)
    Hmat = A + float(damping) * mean_diag * torch.eye(
        6, device=A.device, dtype=A.dtype
    )
    try:
        bread_inv = torch.linalg.inv(Hmat)
    except RuntimeError:
        bread_inv = torch.linalg.pinv(Hmat)

    bs = max(int(block_size), 1)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=r.device),
        torch.arange(W, device=r.device),
        indexing="ij",
    )
    n_blocks_x = (W + bs - 1) // bs
    cluster_id = (
        (yy.reshape(-1) // bs) * n_blocks_x
        + (xx.reshape(-1) // bs)
    )
    cluster_id = cluster_id[active_pixel]
    score_active = score_pixel[active_pixel]

    n_blocks_total = ((H + bs - 1) // bs) * n_blocks_x
    cluster_score = torch.zeros(
        (n_blocks_total, 6), device=r.device, dtype=r.dtype
    )
    cluster_score.index_add_(0, cluster_id, score_active)

    active_cluster = cluster_score.abs().sum(dim=1) > 0
    cluster_score = cluster_score[active_cluster]
    cluster_count = int(cluster_score.shape[0])

    if cluster_count < 2:
        cov = bread_inv
    else:
        meat = cluster_score.T @ cluster_score
        correction = 1.0
        if small_sample_correction and num_pixels > 6:
            correction = (
                cluster_count / float(cluster_count - 1)
            ) * ((num_pixels - 1) / float(num_pixels - 6))
        cov = correction * (bread_inv @ meat @ bread_inv)

    cov = 0.5 * (cov + cov.T)
    eig, vec = torch.linalg.eigh(cov)
    eig = eig.clamp_min(0.0)
    cov = (vec * eig.unsqueeze(0)) @ vec.T
    cov = 0.5 * (cov + cov.T)

    Heig = torch.linalg.eigvalsh(Hmat)
    positive = Heig[Heig > 0]
    if positive.numel() > 0:
        condition = Heig.max() / positive.min().clamp_min(1e-18)
    else:
        condition = torch.tensor(
            float("inf"), device=r.device, dtype=r.dtype
        )

    return {
        "cov": cov,
        "information": Hmat,
        "cluster_count": cluster_count,
        "num_pixels": num_pixels,
        "num_observations": num_observations,
        "condition": condition,
    }


def left_covariance_to_right(T_cw, P_left):
    """Convert left-tangent covariance to right-invariant covariance.

    Exp(eta) T = T Exp(delta), hence delta = Ad_(T^-1) eta.
    """
    A = adjoint_SE3(torch.linalg.inv(T_cw))
    P = A @ P_left @ A.T
    return 0.5 * (P + P.T)


def _stable_information(P, eig_floor_rel=1e-10):
    P = 0.5 * (P + P.T)
    eig, vec = torch.linalg.eigh(P)
    max_eig = eig.max().clamp_min(1e-18)
    eig = eig.clamp_min(max_eig * float(eig_floor_rel))
    return (vec * (1.0 / eig).unsqueeze(0)) @ vec.T


def fuse_prior_tracking_covariance(
    P_prior_right,
    P_track_right,
    information_scale=1.0,
):
    """Information-form fusion of motion prior and tracking observation."""
    I_prior = _stable_information(P_prior_right)
    I_track = _stable_information(P_track_right)
    I_post = I_prior + float(information_scale) * I_track
    try:
        P_post = torch.linalg.inv(I_post)
    except RuntimeError:
        P_post = torch.linalg.pinv(I_post)
    P_post = 0.5 * (P_post + P_post.T)
    eig, vec = torch.linalg.eigh(P_post)
    eig = eig.clamp_min(0.0)
    P_post = (vec * eig.unsqueeze(0)) @ vec.T
    return 0.5 * (P_post + P_post.T)
