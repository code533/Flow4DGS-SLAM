"""M2 pose-to-3D uncertainty propagation utilities.

Conventions:
- Camera pose T_cw is world-to-camera.
- Absolute pose uncertainty uses a right-invariant perturbation:
      T_cw = Tbar_cw Exp(delta_xi^)
  with twist order [rho, phi].
- For X_w = T_cw^{-1} x_c, the pose Jacobian is
      J_pose = [-I, [X_w]_x].
"""

import torch

from utils.pose_utils import SE3_exp, skew_sym_mat


def so3_log_stable(R):
    c = torch.clamp((torch.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.acos(c)
    vee = torch.stack([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    if float(theta.abs()) < 1e-8:
        return 0.5 * vee
    return theta / (2.0 * torch.sin(theta)) * vee


def left_jacobian_so3(w):
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    W = skew_sym_mat(w)
    W2 = W @ W
    theta = torch.linalg.norm(w)
    if float(theta) < 1e-8:
        return I + 0.5 * W + (1.0 / 6.0) * W2
    theta2 = theta * theta
    return (
        I
        + ((1.0 - torch.cos(theta)) / theta2) * W
        + ((theta - torch.sin(theta)) / (theta2 * theta)) * W2
    )


def SE3_log(T):
    w = so3_log_stable(T[:3, :3])
    V = left_jacobian_so3(w)
    t = T[:3, 3]
    try:
        rho = torch.linalg.solve(V, t)
    except RuntimeError:
        rho = torch.linalg.pinv(V) @ t
    return torch.cat([rho, w])


def adjoint_SE3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Z = torch.zeros((3, 3), device=T.device, dtype=T.dtype)
    top = torch.cat([R, skew_sym_mat(t) @ R], dim=1)
    bottom = torch.cat([Z, R], dim=1)
    return torch.cat([top, bottom], dim=0)


def relative_parameter_to_right_jacobian(xi, eps=1e-5):
    """Map M1 exponential-coordinate perturbations to right group errors.

    epsilon(dxi) = Log(Exp(xi)^(-1) Exp(xi + dxi)).
    """
    original_dtype = xi.dtype
    xi64 = xi.detach().to(dtype=torch.float64)
    mean_inv = torch.linalg.inv(SE3_exp(xi64))
    J = torch.empty((6, 6), device=xi64.device, dtype=xi64.dtype)
    for j in range(6):
        e = torch.zeros(6, device=xi64.device, dtype=xi64.dtype)
        e[j] = float(eps)
        ep = SE3_log(mean_inv @ SE3_exp(xi64 + e))
        em = SE3_log(mean_inv @ SE3_exp(xi64 - e))
        J[:, j] = (ep - em) / (2.0 * float(eps))
    return J.to(dtype=original_dtype)


def relative_parameter_cov_to_right(xi, P_xi, eps=1e-5):
    J = relative_parameter_to_right_jacobian(xi, eps=eps)
    P = J @ P_xi @ J.T
    return 0.5 * (P + P.T)


def propagate_right_pose_covariance(P_prev, T_rel, P_rel_right):
    """Propagate covariance for T_k = T_(k-1) DeltaT_k."""
    A = adjoint_SE3(torch.linalg.inv(T_rel))
    P = A @ P_prev @ A.T + P_rel_right
    return 0.5 * (P + P.T)


def backproject_camera_points(uv, depth, K):
    ones = torch.ones((uv.shape[0], 1), device=uv.device, dtype=uv.dtype)
    uv1 = torch.cat([uv, ones], dim=-1)
    rays = (torch.linalg.inv(K) @ uv1.T).T
    return depth[:, None] * rays, rays


def camera_to_world_points(x_c, T_cw):
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    return (R.T @ (x_c - t).T).T


def world_point_covariance(
    uv,
    depth,
    K,
    T_cw,
    P_pose_right,
    depth_var,
    pixel_var=None,
):
    """First-order covariance of RGB-D world points.

    Sigma_X = J_pose P_pose J_pose^T
            + J_depth sigma_d^2 J_depth^T
            + J_uv R_uv J_uv^T.
    """
    x_c, rays = backproject_camera_points(uv, depth, K)
    X_w = camera_to_world_points(x_c, T_cw)
    N = X_w.shape[0]

    I3 = torch.eye(3, device=X_w.device, dtype=X_w.dtype)
    I3 = I3.unsqueeze(0).expand(N, -1, -1)

    X_skew = torch.zeros((N, 3, 3), device=X_w.device, dtype=X_w.dtype)
    x, y, z = X_w.unbind(-1)
    X_skew[:, 0, 1] = -z
    X_skew[:, 0, 2] = y
    X_skew[:, 1, 0] = z
    X_skew[:, 1, 2] = -x
    X_skew[:, 2, 0] = -y
    X_skew[:, 2, 1] = x
    J_pose = torch.cat([-I3, X_skew], dim=-1)

    P_pose = P_pose_right.to(device=X_w.device, dtype=X_w.dtype)
    Sigma_pose = J_pose @ P_pose.unsqueeze(0) @ J_pose.transpose(1, 2)

    R = T_cw[:3, :3]
    J_depth = (R.T @ rays.T).T
    depth_var = torch.as_tensor(depth_var, device=X_w.device, dtype=X_w.dtype)
    if depth_var.ndim == 0:
        depth_var = depth_var.expand(N)
    Sigma_depth = (
        depth_var[:, None, None]
        * J_depth[:, :, None]
        * J_depth[:, None, :]
    )
    Sigma = Sigma_pose + Sigma_depth

    if pixel_var is not None:
        fx = K[0, 0]
        fy = K[1, 1]
        Juv_cam = torch.zeros((N, 3, 2), device=X_w.device, dtype=X_w.dtype)
        Juv_cam[:, 0, 0] = depth / fx
        Juv_cam[:, 1, 1] = depth / fy
        Juv = R.T.unsqueeze(0) @ Juv_cam

        pv = torch.as_tensor(pixel_var, device=X_w.device, dtype=X_w.dtype)
        if pv.ndim == 0:
            Ruv = torch.diag_embed(pv.expand(N, 2))
        elif pv.ndim == 1 and pv.numel() == N:
            Ruv = torch.diag_embed(pv[:, None].expand(-1, 2))
        elif pv.ndim == 1 and pv.numel() == 2:
            Ruv = torch.diag_embed(pv[None].expand(N, -1))
        elif pv.ndim == 2 and pv.shape == (N, 2):
            Ruv = torch.diag_embed(pv)
        else:
            raise ValueError("pixel_var must be scalar, [N], [2], or [N,2]")
        Sigma = Sigma + Juv @ Ruv @ Juv.transpose(1, 2)

    Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2))
    return X_w, Sigma


def covariance_rms_m(Sigma):
    tr = torch.diagonal(Sigma, dim1=-2, dim2=-1).sum(-1)
    return torch.sqrt(torch.clamp(tr / 3.0, min=0.0))
