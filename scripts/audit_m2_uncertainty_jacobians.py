#!/usr/bin/env python3
"""Numerical audit for M2 uncertainty Jacobians."""

import sys
from pathlib import Path

# When executed as `python scripts/<script>.py`, Python places the scripts
# directory (not the repository root) on sys.path. Add the repository root so
# imports such as `utils.m2_uncertainty` work from the documented command.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from utils.m2_uncertainty import (
    adjoint_SE3,
    relative_parameter_to_right_jacobian,
    world_point_covariance,
)
from utils.pose_utils import SE3_exp


def finite_difference_point_jacobian(T_cw, x_c, eps=1e-6):
    J = torch.empty((3, 6), dtype=x_c.dtype)
    one = torch.ones(1, dtype=x_c.dtype)
    for j in range(6):
        e = torch.zeros(6, dtype=x_c.dtype)
        e[j] = eps
        Tp = T_cw @ SE3_exp(e)
        Tm = T_cw @ SE3_exp(-e)
        Xp = (torch.linalg.inv(Tp) @ torch.cat([x_c, one]))[:3]
        Xm = (torch.linalg.inv(Tm) @ torch.cat([x_c, one]))[:3]
        J[:, j] = (Xp - Xm) / (2.0 * eps)
    return J


def main():
    torch.set_default_dtype(torch.float64)

    xi = torch.tensor([0.01, -0.004, 0.006, 0.004, -0.003, 0.002])
    Jr = relative_parameter_to_right_jacobian(xi, eps=1e-6)
    print("xi-to-right Jacobian")
    print(Jr)
    print("||Jr-I||_F =", float(torch.linalg.norm(Jr - torch.eye(6))))

    T_cw = SE3_exp(torch.tensor([0.2, -0.1, 0.3, 0.1, -0.08, 0.05]))
    K = torch.tensor([
        [525.0, 0.0, 319.5],
        [0.0, 525.0, 239.5],
        [0.0, 0.0, 1.0],
    ])
    uv = torch.tensor([[410.0, 210.0]])
    depth = torch.tensor([2.2])
    q = torch.linalg.inv(K) @ torch.tensor([410.0, 210.0, 1.0])
    x_c = depth[0] * q

    J_fd = finite_difference_point_jacobian(T_cw, x_c)
    P = torch.diag(torch.tensor([
        1e-4, 1e-4, 1e-4, 2e-5, 2e-5, 2e-5
    ]))
    Xw, Sigma = world_point_covariance(
        uv, depth, K, T_cw, P,
        depth_var=torch.tensor([4e-4]),
        pixel_var=torch.tensor(0.25),
    )

    X = Xw[0]
    Xhat = torch.tensor([
        [0.0, -X[2], X[1]],
        [X[2], 0.0, -X[0]],
        [-X[1], X[0], 0.0],
    ])
    J_an = torch.cat([-torch.eye(3), Xhat], dim=1)

    jac_err = float(torch.max(torch.abs(J_an - J_fd)))
    sym_err = float(torch.max(torch.abs(Sigma[0] - Sigma[0].T)))
    eig = torch.linalg.eigvalsh(Sigma[0])

    print("\nanalytic J_pose")
    print(J_an)
    print("\nfinite-difference J_pose")
    print(J_fd)
    print("\nmax Jacobian error =", jac_err)
    print("Sigma eigenvalues =", eig.tolist())

    assert jac_err < 1e-5
    assert sym_err < 1e-10
    assert float(eig.min()) >= -1e-12

    eta = torch.tensor([1e-5, -2e-5, 1.5e-5, 2e-5, -1e-5, 0.5e-5])
    lhs = T_cw @ SE3_exp(eta) @ torch.linalg.inv(T_cw)
    rhs = SE3_exp(adjoint_SE3(T_cw) @ eta)
    adj_err = float(torch.max(torch.abs(lhs - rhs)))
    print("adjoint conjugation error =", adj_err)
    assert adj_err < 1e-8

    print("\nM2 uncertainty Jacobian audit PASSED")


if __name__ == "__main__":
    main()
