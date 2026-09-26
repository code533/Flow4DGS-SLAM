import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render, get_dynamic_mask, render_flow
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth, get_loss_network, pearson_loss
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
import os
import matplotlib.pyplot as plt
import random
from utils.slam_backend import (
    segment_motion_parametric,
    fit_twist_weighted,
    fit_twist_probabilistic,
    estimate_fb_flow_variance_px,
    pixels_to_flow_units,
    flow_to_pixels,
)
import torch.nn.functional as F
from utils.pose_utils import SE3_exp, scale_se3_step, SO3_exp, so3_log
from utils.m2_uncertainty import (
    apply_whitened_diag_calibration,
    load_diag6_calibration_report,
    propagate_right_pose_covariance,
    relative_parameter_cov_to_right,
)
from utils.m2_tracking_uncertainty import (
    build_tracking_residual_context,
    cluster_robust_tracking_covariance,
    fuse_prior_tracking_covariance,
    left_covariance_to_right,
    tracking_jacobian_diagnostics,
    tracking_residual_tensor,
)


def flow_to_pixels(flow_norm, H, W, mode='grid'):
    if mode == 'pixel':
        return flow_norm
    if mode == 'grid':
        sx, sy = (W - 1) / 2.0, (H - 1) / 2.0
    elif mode == 'ratio':
        sx, sy = float(W), float(H)
    else:
        raise ValueError("flow_mode must be 'grid' | 'ratio' | 'pixel'")
    scale = torch.tensor([sx, sy], device=flow_norm.device, dtype=flow_norm.dtype)[:, None, None]
    return flow_norm * scale

def build_interaction_matrix(depth, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    y, x = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype),
                          torch.arange(W, device=depth.device, dtype=depth.dtype), indexing='ij')
    u, v = x, y
    Z = depth.clamp_min(1e-6)
    du_dt = torch.stack([-fx/Z, torch.zeros_like(Z), (u-cx)/Z], dim=-1)
    dv_dt = torch.stack([ torch.zeros_like(Z), -fy/Z, (v-cy)/Z], dim=-1)
    du_dw = torch.stack([ (u-cx)*(v-cy)/fy, -(fx + (u-cx)*(u-cx)/fx), (v-cy) ], dim=-1)
    dv_dw = torch.stack([ (fy + (v-cy)*(v-cy)/fy), -(u-cx)*(v-cy)/fx, -(u-cx) ], dim=-1)
    L_top = torch.cat([du_dt, du_dw], dim=-1)
    L_bot = torch.cat([dv_dt, dv_dw], dim=-1)
    L = torch.stack([L_top, L_bot], dim=2).reshape(-1, 6)  # [2HW,6]
    return L

def predict_rigid_flow_px(depth, K, xi):
    H, W = depth.shape
    L = build_interaction_matrix(depth, K)                # [2HW,6]
    fhat = (L @ xi).reshape(H, W, 2).permute(2,0,1)       # [2,H,W] in pixels
    return fhat

def robust_median_residual(flow_meas_px, rigid_flow_px, valid_mask):
    """Median L2 residual over valid pixels."""
    resid = (flow_meas_px - rigid_flow_px).pow(2).sum(0).sqrt()
    rv = resid[valid_mask]
    if rv.numel() == 0:
        return torch.tensor(float('inf'), device=flow_meas_px.device)
    return rv.median()

def warp_mask_from_tminus1(flow_t_to_tm1, mask_tm1, mode='bilinear', padding_mode='zeros'):
    """
    Backward-warp mask at t-1 into frame t using flow from t -> t-1.

    Args:
        flow_t_to_tm1: Tensor of shape [H, W, 2] or [B, H, W, 2], pixel units (u,v).
                       u is x-displacement (cols), v is y-displacement (rows).
        mask_tm1:     Tensor of shape [H, W], [1, H, W], or [B, 1, H, W].
                      Values will be sampled from this (e.g., {0,1} motion mask at t-1).
        mode:         'bilinear' (default) or 'nearest' for sampling.
        padding_mode: 'zeros' | 'border' | 'reflection' (what to do when sampling outside).
    Returns:
        mask_t: Tensor of shape [H, W] (if inputs unbatched) or [B, 1, H, W] (if batched).
    """
    # --- Normalize to batched shapes ---
    batched = flow_t_to_tm1.dim() == 4
    if not batched:
        flow = flow_t_to_tm1.unsqueeze(0)       # [1,H,W,2]
    else:
        flow = flow_t_to_tm1                     # [B,H,W,2]

    if mask_tm1.dim() == 2:
        mask = mask_tm1.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif mask_tm1.dim() == 3:  # [1,H,W] -> [1,1,H,W]
        mask = mask_tm1.unsqueeze(0) if mask_tm1.size(0) != 1 else mask_tm1.unsqueeze(1)
        if mask.dim() != 4:
            mask = mask_tm1.unsqueeze(0)
    else:
        mask = mask_tm1  # assume [B,1,H,W]

    B, H, W = flow.size(0), flow.size(1), flow.size(2)
    device = flow.device
    dtype = flow.dtype

    # --- Build base grid in normalized coords [-1,1] ---
    # grid_sample expects grid[..., 0] = x in [-1,1], grid[..., 1] = y in [-1,1]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    base_x = (xx / (W - 1)) * 2 - 1  # [H,W]
    base_y = (yy / (H - 1)) * 2 - 1  # [H,W]
    base_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0).expand(B, H, W, 2)  # [B,H,W,2]

    # --- Convert pixel flow to normalized offsets and add to base grid ---
    # flow[...,0] = u (x), flow[...,1] = v (y) in pixels
    norm_flow = torch.empty_like(flow)
    norm_flow[..., 0] = flow[..., 0] * (2.0 / max(W - 1, 1))
    norm_flow[..., 1] = flow[..., 1] * (2.0 / max(H - 1, 1))
    grid = base_grid + norm_flow  # sampling locations in frame t-1 for each pixel in frame t

    # Ensure mask has batch & channel
    if mask.dim() == 3:  # [B,H,W] -> [B,1,H,W]
        mask = mask.unsqueeze(1)
    if mask.size(0) != B:
        # If a single mask is provided but multiple flows, broadcast it
        if mask.size(0) == 1:
            mask = mask.expand(B, -1, -1, -1)
        else:
            raise ValueError("Batch size of mask and flow must match, or mask batch must be 1.")

    # --- Sample: this performs the backward warp ---
    mask_t = F.grid_sample(
        mask, grid, mode=mode, align_corners=True, padding_mode=padding_mode
    )  # [B,1,H,W]

    # Squeeze if inputs were unbatched
    if not batched:
        return mask_t[0, 0]  # [H,W]
    return mask_t  # [B,1,H,W]

def _skew(omega):
    wx, wy, wz = omega[...,0], omega[...,1], omega[...,2]
    O = torch.zeros((*omega.shape[:-1], 3, 3), device=omega.device, dtype=omega.dtype)
    O[..., 0,1] = -wz; O[..., 0,2] =  wy
    O[..., 1,0] =  wz; O[..., 1,2] = -wx
    O[..., 2,0] = -wy; O[..., 2,1] =  wx
    return O




def vis_render_process(gaussians, pipeline_params, background, viewpoint, cur_frame_idx, save_dir, out_dir="track", mask=None):
    with torch.no_grad():
        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background, mask=mask)
        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)
        
        fig, ax = plt.subplots(figsize=(8, 8))  # the size of the figure
        cax = ax.imshow(viz_im)
        ax.axis('off')
        # save the figure
        os.makedirs(save_dir, exist_ok=True)
        process_dir = os.path.join(save_dir, out_dir)
        os.makedirs(process_dir, exist_ok=True)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        return
        

        

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        self.dynamic_model = config["model_params"].get("dynamic_model", False)
        self.insertion = config["model_params"].get("insertion", False)
        self.flow_mask = config["model_params"].get("flow_mask", False)
        self.flow_cam = config["model_params"].get("flow_cam", False)
        self.flow_cam_thres = config["model_params"].get("flow_cam_thres", 5.0)
        self.flow_cam_maxt = config["model_params"].get("flow_cam_maxt", 0.01)
        self.flow_cam_maxr = config["model_params"].get("flow_cam_maxr", 0.1)

        # M1: local probabilistic camera-motion estimation. The switch is
        # deliberately independent of the baseline flow mask so M1 can be
        # enabled/disabled without changing the rest of the pipeline.
        unc_cfg = config.get("Uncertainty", {})
        self.m1_uncertainty = bool(unc_cfg.get("enable_m1", False))
        self.m1_flow_sigma0_px = float(unc_cfg.get("flow_sigma0_px", 0.5))
        self.m1_flow_fb_lambda = float(unc_cfg.get("flow_fb_lambda", 1.0))
        self.m1_flow_max_sigma_px = float(unc_cfg.get("flow_max_sigma_px", 20.0))
        self.m1_depth_sigma0_m = float(unc_cfg.get("depth_sigma0_m", 0.01))
        self.m1_depth_sigma_rel = float(unc_cfg.get("depth_sigma_rel", 0.0))
        self.m1_cauchy_c = float(unc_cfg.get("cauchy_c", 2.0))
        self.m1_damping = float(unc_cfg.get("damping", 1e-6))
        self.m1_min_pixels = int(unc_cfg.get("min_pixels", 500))
        self.m1_residual_rescale = bool(unc_cfg.get("residual_rescale", False))
        self.m1_save_diagnostics = bool(unc_cfg.get("save_diagnostics", True))
        self.m1_cluster_covariance = bool(unc_cfg.get("cluster_covariance", True))
        self.m1_cluster_block_size = int(unc_cfg.get("cluster_block_size", 32))
        self.m1_cluster_block_sizes = [
            int(v) for v in unc_cfg.get("cluster_block_sizes", [16, 32, 64])
        ]
        self.m1_cluster_small_sample = bool(
            unc_cfg.get("cluster_small_sample_correction", True)
        )
        self.m1_shadow_mode = bool(unc_cfg.get("shadow_mode", True))
        self.m1_keyframe_reason_log = bool(
            unc_cfg.get("log_keyframe_reasons", True)
        )

        # M2-A: shadow-only propagation of absolute camera-pose uncertainty.
        # It does not alter the baseline pose mean, losses, keyframes, mapping,
        # densification, or deformation optimization.
        self.m2a_enable = bool(unc_cfg.get("enable_m2a", False))
        self.m2a_save_diagnostics = bool(
            unc_cfg.get("m2a_save_diagnostics", True)
        )
        self.m2a_use_diag_calibration = bool(
            unc_cfg.get("m2a_use_diag_calibration", False)
        )
        self.m2a_diag_calibration = unc_cfg.get(
            "m2a_diag_calibration", None
        )
        self.m2a_diag_calibration_file = unc_cfg.get(
            "m2a_diag_calibration_file", None
        )
        self.m2a_diag_calibration_fold = unc_cfg.get(
            "m2a_diag_calibration_fold", None
        )
        self.m2a_diag_calibration_source = None
        self.m2a_diag_calibration_train_sequences = []

        if self.m2a_use_diag_calibration:
            if self.m2a_diag_calibration is not None:
                values = torch.as_tensor(
                    self.m2a_diag_calibration, dtype=torch.float64
                ).reshape(-1)
                if values.numel() != 6:
                    raise ValueError(
                        "Uncertainty.m2a_diag_calibration must contain "
                        "exactly 6 positive values."
                    )
                if (
                    not bool(torch.isfinite(values).all())
                    or bool((values <= 0).any())
                ):
                    raise ValueError(
                        "Uncertainty.m2a_diag_calibration must be finite "
                        "and strictly positive."
                    )
                self.m2a_diag_calibration = values.tolist()
                self.m2a_diag_calibration_source = "inline-config"
            elif self.m2a_diag_calibration_file:
                calib_info = load_diag6_calibration_report(
                    self.m2a_diag_calibration_file,
                    fold=self.m2a_diag_calibration_fold,
                )
                report_block = int(calib_info["block_size"])
                if (
                    report_block > 0
                    and report_block != self.m1_cluster_block_size
                ):
                    raise ValueError(
                        "M2-A1 calibration block-size mismatch: report uses "
                        f"{report_block}, but M1 cluster_block_size is "
                        f"{self.m1_cluster_block_size}."
                    )
                self.m2a_diag_calibration = calib_info["diag"]
                self.m2a_diag_calibration_source = calib_info["source"]
                self.m2a_diag_calibration_train_sequences = calib_info[
                    "train_sequences"
                ]
                Log(
                    "M2-A1 loaded Diag-6 calibration",
                    self.m2a_diag_calibration,
                    "source", self.m2a_diag_calibration_source,
                    "train", self.m2a_diag_calibration_train_sequences,
                    tag="Frontend",
                )
            else:
                raise ValueError(
                    "m2a_use_diag_calibration=true requires either "
                    "m2a_diag_calibration=[c1,...,c6] or "
                    "m2a_diag_calibration_file=<M1 calibration JSON>."
                )

        self.m2a_right_jacobian_eps = float(
            unc_cfg.get("m2a_right_jacobian_eps", 1.0e-5)
        )

        # M2-A2: tracking-posterior covariance audit. This is intentionally
        # disabled by default because finite-difference rendering adds twelve
        # extra static renders per audited frame.
        self.m2a2_enable = bool(unc_cfg.get("enable_m2a2", False))
        self.m2a2_save_diagnostics = bool(
            unc_cfg.get("m2a2_save_diagnostics", True)
        )
        self.m2a2_block_size = int(
            unc_cfg.get("m2a2_cluster_block_size", 32)
        )
        self.m2a2_trans_eps = float(
            unc_cfg.get("m2a2_translation_eps", 1.0e-4)
        )
        self.m2a2_rot_eps = float(
            unc_cfg.get("m2a2_rotation_eps", 1.0e-4)
        )
        self.m2a2_cauchy_c = float(
            unc_cfg.get("m2a2_cauchy_c", 2.0)
        )
        self.m2a2_damping = float(
            unc_cfg.get("m2a2_damping", 1.0e-6)
        )
        self.m2a2_information_scale = float(
            unc_cfg.get("m2a2_information_scale", 1.0)
        )
        self.m2a2_rgb_scale_floor = float(
            unc_cfg.get("m2a2_rgb_scale_floor", 0.01)
        )
        self.m2a2_depth_scale_floor = float(
            unc_cfg.get("m2a2_depth_scale_floor", 0.01)
        )

        # Optional finite-difference sensitivity audit. When enabled, only
        # selected frames are swept so we can diagnose rasterizer resolution
        # without multiplying the full-sequence runtime excessively.
        self.m2a2_fd_audit = bool(
            unc_cfg.get("m2a2_fd_audit", False)
        )
        self.m2a2_fd_audit_frames = set(
            int(v) for v in unc_cfg.get(
                "m2a2_fd_audit_frames", [50, 200, 400, 600]
            )
        )
        self.m2a2_fd_translation_scales = [
            float(v) for v in unc_cfg.get(
                "m2a2_fd_translation_scales",
                [1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3],
            )
        ]
        self.m2a2_fd_rotation_scales = [
            float(v) for v in unc_cfg.get(
                "m2a2_fd_rotation_scales",
                [1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3],
            )
        ]

        if self.m2a2_enable and not self.m2a_enable:
            raise ValueError(
                "enable_m2a2=true requires enable_m2a=true because the "
                "tracking posterior needs an M2-A motion prior."
            )

        if self.m2a_enable and not self.m1_uncertainty:
            Log(
                "M2-A requested while M1 is disabled; M2-A propagation will "
                "remain invalid until an M1 relative covariance is available.",
                tag="Frontend",
            )

        self.dynamic_objects = 0

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0 
            return initial_depth.cpu().numpy()[0]
        
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0
        if self.dynamic_model:
            initial_depth = initial_depth.detach().clone()  
            if (not self.insertion) or init:
                initial_depth[0][~viewpoint.motion_mask.cpu().numpy()] = 0
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        print('SETTING GT POSE DURING INITIALIZATION')

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        if self.m2a_enable:
            # Frame 0 is explicitly initialized from GT by the baseline code,
            # so its shadow covariance is anchored at zero.
            viewpoint.pose_cov_abs_right.zero_()
            viewpoint.pose_cov_abs_right_raw.zero_()
            viewpoint.pose_cov_rel_right = torch.zeros(
                (6, 6),
                device=viewpoint.R.device,
                dtype=viewpoint.R.dtype,
            )
            viewpoint.pose_cov_valid = True
            viewpoint.pose_cov_source = "gt_anchor"

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

            

    def tracking(self, cur_frame_idx, viewpoint, last_keyframe_idx):
        start_time = time.time()
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        
        viewpoint.update_RT(prev.R, prev.T)
        m2a_prior_pose = None
        m2a_rel_right = None
        m2a_abs_raw = None
        m2a_abs_cal = None
        m2a2_track_left = None
        m2a2_track_right = None
        m2a2_post_right = None
        m2a2_stats = None
        m2a2_fd_audit_payload = None
        
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )
        
        pose_optimizer = torch.optim.Adam(opt_params, capturable=True)
        
        with torch.no_grad():
            dxyz, d_rot, d_scale = None, None, None
        
        output_dir = os.path.join(self.config["Results"]["save_dir"], "tracking")
        os.makedirs(output_dir, exist_ok=True)
        initial_motion_mask = viewpoint.motion_mask.clone()

        if self.flow_mask:
            if self.save_results:
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, mask=(self.gaussians.dygs==False))

            if self.m1_uncertainty:
                # M1 needs both flow directions for forward/backward
                # consistency. Camera.generate_flow(..., tracking=False)
                # returns (previous->current, current->previous) in the
                # repository's normalized grid-flow units.
                flow_fwd, flow_back = viewpoint.generate_flow(
                    viewpoint.original_image.cuda(),
                    prev.original_image.cuda(),
                    tracking=False,
                    cache=False,
                )
            else:
                flow_back = viewpoint.generate_flow(
                    viewpoint.original_image.cuda(),
                    prev.original_image.cuda(),
                    tracking=True,
                )
                flow_fwd = None
            
            depth_tensor = torch.from_numpy(viewpoint.depth).cuda().squeeze().type(torch.float32)
            H, W = depth_tensor.shape
            results = segment_motion_parametric(
                                    depth_tensor, 
                                    flow_back.permute(2,0,1), 
                                    viewpoint.intrinsic.cuda(),
                                    flow_mode='grid',
                                    ds=1,
                                    robust_iters=30,
                                    k_mad=self.flow_cam_thres,
                                    motion_mask=viewpoint.motion_mask.cuda()
                                )
            flow_mask = results['mask_bool']
            
            viewpoint.motion_mask = viewpoint.motion_mask & (~flow_mask)

            rigid_flow = results['rigid_flow_out'].permute(1,2,0)

            if self.save_results:
                value_np = results['resid_px'].detach().cpu().numpy()

                plt.figure(figsize=(6, 4))
                plt.imshow(value_np, cmap='viridis', origin='upper')
                plt.colorbar(label='Value')
                plt.xlabel('W')
                plt.ylabel('H')
                plt.title('Value Map Heatmap')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"tracking_{viewpoint.uid}_resid_heatmap.png"))


            if self.flow_cam:

                xi = results['xi']                                # [6], on CUDA/float32 already

                depth_valid = (depth_tensor > 0) & torch.isfinite(depth_tensor)

                static_prior = initial_motion_mask.cuda().bool() if 'initial_motion_mask' in locals() else torch.ones_like(depth_valid, dtype=torch.bool)

                static_from_residual = ~results['mask_bool']
                static_refit_mask = depth_valid & static_prior & static_from_residual  
                valid_extent = static_refit_mask.sum().item() / static_refit_mask.numel()
                static_inliers_ds = F.interpolate(static_refit_mask[None,None].float(), size=(H,W), mode='nearest').squeeze().bool()
                
                flow_px = flow_to_pixels(flow_back.permute(2,0,1), H, W, mode='grid')
                depth_valid = (depth_tensor > 0) & torch.isfinite(depth_tensor)
                depth_ds, depth_valid_ds, flow_px_ds, Kds = depth_tensor, depth_valid, flow_px, viewpoint.intrinsic.cuda()
                
                # Preserve the original Flow4DGS pose mean exactly. M1 can
                # evaluate uncertainty in strict shadow mode without changing
                # the state trajectory used by tracking, mapping, or keyframe
                # selection.
                xi_baseline = fit_twist_weighted(
                    depth_ds,
                    flow_px_ds,
                    Kds,
                    static_inliers_ds,
                    robust=True,
                    iters=30,
                )

                m1_result = None
                fb_error_px = None
                static_prob_mask = static_inliers_ds

                if self.m1_uncertainty:
                    # The pose refit currently runs at full image resolution,
                    # so the FB uncertainty is constructed at the same scale.
                    flow_fwd_px = flow_to_pixels(
                        flow_fwd.permute(2,0,1), H, W, mode='grid'
                    )

                    flow_var_px2, fb_valid, fb_error_px = estimate_fb_flow_variance_px(
                        flow_px_ds,
                        flow_fwd_px,
                        sigma0_px=self.m1_flow_sigma0_px,
                        lambda_fb=self.m1_flow_fb_lambda,
                        max_sigma_px=self.m1_flow_max_sigma_px,
                    )

                    static_prob_mask = static_inliers_ds & fb_valid

                    # M1 Version 1 uses a deliberately simple depth-noise
                    # model. It can later be replaced by a calibrated RGB-D
                    # sensor model without changing the GLS solver.
                    depth_sigma = (
                        self.m1_depth_sigma0_m
                        + self.m1_depth_sigma_rel * depth_ds.abs()
                    )
                    depth_var_m2 = depth_sigma.square()

                    m1_result = fit_twist_probabilistic(
                        depth_ds,
                        flow_px_ds,
                        Kds,
                        static_prob_mask,
                        flow_var_px2,
                        depth_var_m2,
                        robust=True,
                        iters=10,
                        cauchy_c=self.m1_cauchy_c,
                        damping=self.m1_damping,
                        residual_rescale=self.m1_residual_rescale,
                        min_pixels=self.m1_min_pixels,
                        cluster_covariance=self.m1_cluster_covariance,
                        cluster_block_size=self.m1_cluster_block_size,
                        cluster_block_sizes=self.m1_cluster_block_sizes,
                        cluster_small_sample_correction=self.m1_cluster_small_sample,
                        fixed_xi=xi_baseline if self.m1_shadow_mode else None,
                    )
                    xi = xi_baseline if self.m1_shadow_mode else m1_result["xi"]
                    viewpoint.pose_cov_rel_raw = m1_result["cov"].detach().cpu()
                    viewpoint.pose_cov_rel_hessian = m1_result[
                        "cov_hessian"
                    ].detach().cpu()
                    viewpoint.pose_cov_rel_cluster = m1_result[
                        "cov_cluster"
                    ].detach().cpu()
                else:
                    xi = xi_baseline

                rigid_flow_px = predict_rigid_flow_px(depth_ds, Kds, xi)
                rigid_flow_out = pixels_to_flow_units(rigid_flow_px, H, W, mode='grid')

                rigid_flow_out[:, ~static_inliers_ds] = 0.0

                rigid_flow_out = rigid_flow_out.permute(1,2,0)
                
            
                T_rel = SE3_exp(xi)                        
                R_prev = prev.R.cuda().float()
                t_prev = prev.T.cuda().float()
                T_prev = torch.eye(4, device=R_prev.device, dtype=R_prev.dtype)
                T_prev[:3,:3] = R_prev
                T_prev[:3,  3] = t_prev

                T_new = T_prev @ T_rel                         

                T_curr = T_new

                max_translation = self.flow_cam_maxt * valid_extent  
                max_rot_deg = self.flow_cam_maxr * valid_extent      
                max_rot = torch.deg2rad(torch.tensor(max_rot_deg, device=T_curr.device))

                delta_t = T_curr[:3, 3] - T_prev[:3, 3]
                delta_R = T_curr[:3, :3] @ T_prev[:3, :3].T

                trace = torch.clamp((torch.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)
                angle = torch.acos(trace)

                trans_norm = torch.norm(delta_t)
                s_t = max_translation / (trans_norm + 1e-10)

                s_r = max_rot / (angle + 1e-10)

                s = torch.clamp(torch.minimum(s_t, s_r), max=1.0)

                delta_t_scaled = s_t * delta_t

                w = so3_log(delta_R)
                delta_R_scaled = SO3_exp(s * w) 

                T_curr[:3, :3] = delta_R_scaled @ T_prev[:3, :3]
                T_curr[:3, 3]  = T_prev[:3, 3] + delta_t_scaled

                viewpoint.R = T_curr[:3,:3]
                viewpoint.T = T_curr[:3, 3]

                if (
                    self.m2a_enable
                    and self.m1_uncertainty
                    and m1_result is not None
                ):
                    # M1 returns covariance of the exponential-coordinate
                    # parameter xi. Convert it to right-invariant relative
                    # group-error covariance before recursive propagation.
                    P_rel_param_raw = m1_result["cov_cluster"]

                    # M1 Diag-6 was fitted in the xi-parameter covariance
                    # space, so calibration must be applied before converting
                    # to a right-invariant group-error covariance.
                    if (
                        self.m2a_use_diag_calibration
                        and self.m2a_diag_calibration is not None
                    ):
                        P_rel_param_cal = apply_whitened_diag_calibration(
                            P_rel_param_raw,
                            self.m2a_diag_calibration,
                        )
                    else:
                        P_rel_param_cal = P_rel_param_raw

                    m2a_rel_right_raw = relative_parameter_cov_to_right(
                        xi,
                        P_rel_param_raw,
                        eps=self.m2a_right_jacobian_eps,
                    )
                    m2a_rel_right = relative_parameter_cov_to_right(
                        xi,
                        P_rel_param_cal,
                        eps=self.m2a_right_jacobian_eps,
                    )

                    P_prev_abs = prev.pose_cov_abs_right.to(
                        device=T_rel.device, dtype=T_rel.dtype
                    )
                    P_prev_raw = prev.pose_cov_abs_right_raw.to(
                        device=T_rel.device, dtype=T_rel.dtype
                    )

                    # The baseline applies a deterministic motion cap after
                    # the raw flow estimate. Use the actually applied relative
                    # mean for adjoint transport, while retaining the M1
                    # covariance as the process-noise approximation. The cap
                    # correction is logged for later audit.
                    T_rel_applied_m2a = torch.linalg.inv(T_prev) @ T_curr

                    if prev.pose_cov_valid:
                        m2a_abs_raw = propagate_right_pose_covariance(
                            P_prev_raw,
                            T_rel_applied_m2a,
                            m2a_rel_right_raw,
                        )
                    else:
                        # If a previous frame has no valid propagated state,
                        # restart the shadow chain from the current relative
                        # motion rather than injecting fabricated certainty.
                        m2a_abs_raw = m2a_rel_right_raw.clone()

                    if prev.pose_cov_valid:
                        m2a_abs_cal = propagate_right_pose_covariance(
                            P_prev_abs,
                            T_rel_applied_m2a,
                            m2a_rel_right,
                        )
                    else:
                        m2a_abs_cal = m2a_rel_right.clone()

                    viewpoint.pose_cov_rel_right = (
                        m2a_rel_right.detach().clone()
                    )
                    viewpoint.pose_cov_abs_right_raw = (
                        m2a_abs_raw.detach().clone()
                    )
                    viewpoint.pose_cov_abs_right = (
                        m2a_abs_cal.detach().clone()
                    )
                    viewpoint.pose_cov_valid = True
                    viewpoint.pose_cov_source = (
                        "cluster32+diag6"
                        if (
                            self.m2a_use_diag_calibration
                            and self.m2a_diag_calibration is not None
                        )
                        else "cluster32_raw"
                    )

                    # Store the pose mean at which the motion prior was
                    # propagated. Subsequent photometric tracking re-centres
                    # the pose mean but does not change shadow covariance.
                    m2a_prior_pose = torch.eye(
                        4, device=T_curr.device, dtype=T_curr.dtype
                    )
                    m2a_prior_pose[:3, :3] = viewpoint.R
                    m2a_prior_pose[:3, 3] = viewpoint.T

                if self.m1_uncertainty and m1_result is not None:
                    # Keep the covariance tied to the raw GLS relative-pose
                    # estimate. The subsequent Flow4DGS motion cap is a
                    # baseline heuristic and is logged separately.
                    P_xi = m1_result["cov"]
                    P_hessian = m1_result["cov_hessian"]
                    P_cluster = m1_result["cov_cluster"]
                    diag = torch.diagonal(P_xi)
                    sigma_trans = torch.sqrt(diag[:3].clamp_min(0.0))
                    sigma_rot = torch.sqrt(diag[3:].clamp_min(0.0))

                    if self.m1_save_diagnostics:
                        m1_dir = os.path.join(
                            self.config["Results"]["save_dir"],
                            "m1_pose_uncertainty",
                        )
                        os.makedirs(m1_dir, exist_ok=True)

                        if static_prob_mask.any():
                            fb_median = fb_error_px[static_prob_mask].median()
                            maha_median = m1_result["maha_map"][static_prob_mask].median()
                        else:
                            fb_median = torch.tensor(
                                float("nan"), device=depth_ds.device, dtype=depth_ds.dtype
                            )
                            maha_median = torch.tensor(
                                float("nan"), device=depth_ds.device, dtype=depth_ds.dtype
                            )

                        T_rel_applied = torch.linalg.inv(T_prev) @ T_curr

                        # Ground-truth relative pose in the same right-composed
                        # convention used by the raw Flow4DGS update:
                        # T_curr = T_prev @ T_rel.
                        T_prev_gt = torch.eye(
                            4, device=T_prev.device, dtype=T_prev.dtype
                        )
                        T_prev_gt[:3, :3] = prev.R_gt.to(
                            device=T_prev.device, dtype=T_prev.dtype
                        )
                        T_prev_gt[:3, 3] = prev.T_gt.to(
                            device=T_prev.device, dtype=T_prev.dtype
                        )

                        T_curr_gt = torch.eye(
                            4, device=T_prev.device, dtype=T_prev.dtype
                        )
                        T_curr_gt[:3, :3] = viewpoint.R_gt.to(
                            device=T_prev.device, dtype=T_prev.dtype
                        )
                        T_curr_gt[:3, 3] = viewpoint.T_gt.to(
                            device=T_prev.device, dtype=T_prev.dtype
                        )

                        T_rel_gt = torch.linalg.inv(T_prev_gt) @ T_curr_gt

                        payload = {
                            "frame": int(viewpoint.uid),
                            "xi_raw": xi.detach().cpu(),
                            "xi_baseline": xi_baseline.detach().cpu(),
                            "xi_probabilistic": m1_result["xi"].detach().cpu(),
                            "shadow_mode": bool(self.m1_shadow_mode),
                            "P_xi_raw": P_xi.detach().cpu(),
                            "P_xi_hessian": P_hessian.detach().cpu(),
                            "P_xi_cluster": P_cluster.detach().cpu(),
                            "P_xi_clusters": {
                                int(bs): P.detach().cpu()
                                for bs, P in m1_result["cov_clusters"].items()
                            },
                            "covariance_mode": m1_result["covariance_mode"],
                            "cluster_count": int(m1_result["cluster_count"]),
                            "cluster_counts": {
                                int(bs): int(n)
                                for bs, n in m1_result["cluster_counts"].items()
                            },
                            "cluster_block_size": int(m1_result["cluster_block_size"]),
                            "cluster_block_sizes": [
                                int(bs) for bs in m1_result["cluster_block_sizes"]
                            ],
                            "T_rel_raw": T_rel.detach().cpu(),
                            "T_rel_applied": T_rel_applied.detach().cpu(),
                            "T_rel_gt": T_rel_gt.detach().cpu(),
                            "sigma_trans_m": sigma_trans.detach().cpu(),
                            "sigma_rot_rad": sigma_rot.detach().cpu(),
                            "kappa": m1_result["kappa"].detach().cpu(),
                            "condition": m1_result["condition"].detach().cpu(),
                            "num_pixels": int(m1_result["num_pixels"]),
                            "valid_extent": float(valid_extent),
                            "fb_error_median_px": fb_median.detach().cpu(),
                            "maha_median": maha_median.detach().cpu(),
                            "motion_scale_joint": s.detach().cpu(),
                            "motion_scale_translation": s_t.detach().cpu(),
                        }
                        torch.save(
                            payload,
                            os.path.join(m1_dir, f"{int(viewpoint.uid):06d}.pt"),
                        )

                    Log(
                        "M1 frame", viewpoint.uid,
                        "sigma_t", sigma_trans.detach().cpu().tolist(),
                        "sigma_r", sigma_rot.detach().cpu().tolist(),
                        "kappa", float(m1_result["kappa"].detach().cpu()),
                        "cond", float(m1_result["condition"].detach().cpu()),
                        "cov", m1_result["covariance_mode"],
                        "shadow", bool(self.m1_shadow_mode),
                        "clusters", int(m1_result["cluster_count"]),
                        "block", int(m1_result["cluster_block_size"]),
                        "blocks_all", [
                            int(bs) for bs in m1_result["cluster_block_sizes"]
                        ],
                        "N", int(m1_result["num_pixels"]),
                        tag="Frontend",
                    )



            # print('flow mask tracking time:', time.time() - start_time)
            if self.save_results:# and (cur_frame_idx-last_keyframe_idx==5):
                fig, axes = plt.subplots(1, 4, figsize=(15, 5))
                axes[0].imshow(viewpoint.original_image.permute(1, 2, 0).cpu().numpy())
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                axes[1].imshow(~initial_motion_mask.cpu().numpy(), cmap='gray')
                axes[1].set_title("Initial Motion Mask")
                axes[1].axis("off")

                axes[2].imshow(flow_mask.cpu().numpy(), cmap='gray')
                axes[2].set_title("Refined Flow-based Motion Mask")
                axes[2].axis("off")

                axes[3].imshow((~viewpoint.motion_mask).cpu().numpy(), cmap='gray')
                axes[3].set_title("Final Motion Mask")
                axes[3].axis("off")

                output_dir = os.path.join(self.config["Results"]["save_dir"], "tracking")
                os.makedirs(output_dir, exist_ok=True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'tracking_{viewpoint.uid}_flowmask.png'), dpi=150)
                plt.close()

        loss_tracking_init = 0.0

        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, mask=(self.gaussians.dygs==False),)
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            
            # remove the dynamic object at frame 0
            with torch.no_grad():
                if self.dynamic_model and self.gaussians.deform_init and False:
                    mask = viewpoint.reproject_mask(self.dataset, self.cameras[last_keyframe_idx])
                else:
                    mask = None


            loss_tracking = get_loss_tracking(
                self.config, 
                image, 
                depth, 
                opacity, 
                viewpoint, 
                rm_dynamic=True, 
                mask=mask, #not self.dynamic_model
                save_img = False,
            )

            if tracking_itr == 0:
                loss_tracking_init = loss_tracking.item()


            

            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                pose_optimizer.zero_grad()
                if self.gaussians.init_deform == 'mlp':
                    self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                converged = update_pose(viewpoint)

            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)

        # M2-A2: estimate a cluster-robust tracking pose covariance around
        # the FINAL tracked pose and fuse it with the M2-A motion prior.
        if (
            self.m2a2_enable
            and self.m2a_enable
            and viewpoint.pose_cov_valid
            and m2a_prior_pose is not None
        ):
            with torch.no_grad():
                # The tracking loop optimizes only static Gaussians.
                static_mask = (self.gaussians.dygs == False)
                final_static_pkg = render(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    dynamic=False,
                    dx=dxyz,
                    ds=d_scale,
                    dr=d_rot,
                    mask=static_mask,
                )
                track_context = build_tracking_residual_context(
                    self.config,
                    final_static_pkg,
                    viewpoint,
                    rm_dynamic=True,
                    extra_mask=None,
                    rgb_scale_floor=self.m2a2_rgb_scale_floor,
                    depth_scale_floor=self.m2a2_depth_scale_floor,
                )
                residual0, valid0 = tracking_residual_tensor(
                    final_static_pkg, viewpoint, track_context
                )

                C, H, W = residual0.shape
                J_track = torch.empty(
                    (C, H, W, 6),
                    device=residual0.device,
                    dtype=residual0.dtype,
                )

                # IMPORTANT: cam_trans_delta/cam_rot_delta are used by
                # the custom rasterizer as backward pose-gradient hooks; their
                # values do not perturb the forward render. Finite differences
                # must therefore perturb the actual camera pose mean.
                #
                # The baseline tracking update convention is left-multiplicative:
                #     T' = Exp(delta) @ T.
                # We use the same convention here and restore the exact final
                # tracked pose after every render.
                viewpoint.cam_trans_delta.data.zero_()
                viewpoint.cam_rot_delta.data.zero_()

                R_fd_base = viewpoint.R.detach().clone()
                t_fd_base = viewpoint.T.detach().clone()
                T_fd_base = torch.eye(
                    4,
                    device=R_fd_base.device,
                    dtype=R_fd_base.dtype,
                )
                T_fd_base[:3, :3] = R_fd_base
                T_fd_base[:3, 3] = t_fd_base

                eps = [
                    self.m2a2_trans_eps,
                    self.m2a2_trans_eps,
                    self.m2a2_trans_eps,
                    self.m2a2_rot_eps,
                    self.m2a2_rot_eps,
                    self.m2a2_rot_eps,
                ]
                for j in range(6):
                    delta = torch.zeros(
                        6,
                        device=T_fd_base.device,
                        dtype=T_fd_base.dtype,
                    )
                    delta[j] = float(eps[j])
                    T_p = SE3_exp(delta) @ T_fd_base
                    viewpoint.update_RT(T_p[:3, :3], T_p[:3, 3])

                    pkg_p = render(
                        viewpoint,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                        dynamic=False,
                        dx=dxyz,
                        ds=d_scale,
                        dr=d_rot,
                        mask=static_mask,
                    )
                    r_p, _ = tracking_residual_tensor(
                        pkg_p, viewpoint, track_context
                    )

                    delta[j] = -float(eps[j])
                    T_m = SE3_exp(delta) @ T_fd_base
                    viewpoint.update_RT(T_m[:3, :3], T_m[:3, 3])

                    pkg_m = render(
                        viewpoint,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                        dynamic=False,
                        dx=dxyz,
                        ds=d_scale,
                        dr=d_rot,
                        mask=static_mask,
                    )
                    r_m, _ = tracking_residual_tensor(
                        pkg_m, viewpoint, track_context
                    )

                    J_track[..., j] = (r_p - r_m) / (2.0 * eps[j])
                    viewpoint.update_RT(R_fd_base, t_fd_base)

                viewpoint.update_RT(R_fd_base, t_fd_base)

                jac_diag = tracking_jacobian_diagnostics(
                    residual0,
                    J_track,
                    valid0,
                    cauchy_c=self.m2a2_cauchy_c,
                    damping=self.m2a2_damping,
                )

                # Optional epsilon sweep on a sparse frame subset. Each scale
                # evaluates all 6 DoF by central difference. The resulting
                # Jacobian magnitudes and bread spectra reveal whether the
                # nominal 1e-4 step sits below the rasterizer's useful
                # numerical resolution.
                if (
                    self.m2a2_fd_audit
                    and int(viewpoint.uid) in self.m2a2_fd_audit_frames
                ):
                    sweep = {}
                    for family, values in [
                        ("translation", self.m2a2_fd_translation_scales),
                        ("rotation", self.m2a2_fd_rotation_scales),
                    ]:
                        family_rows = []
                        for audit_eps in values:
                            J_audit = torch.empty_like(J_track)
                            diff_rms = torch.zeros(
                                6,
                                device=residual0.device,
                                dtype=residual0.dtype,
                            )
                            for j in range(6):
                                step = (
                                    float(audit_eps)
                                    if (
                                        (family == "translation" and j < 3)
                                        or (family == "rotation" and j >= 3)
                                    )
                                    else (
                                        self.m2a2_trans_eps
                                        if j < 3
                                        else self.m2a2_rot_eps
                                    )
                                )

                                delta = torch.zeros(
                                    6,
                                    device=T_fd_base.device,
                                    dtype=T_fd_base.dtype,
                                )
                                delta[j] = float(step)
                                T_p = SE3_exp(delta) @ T_fd_base
                                viewpoint.update_RT(
                                    T_p[:3, :3], T_p[:3, 3]
                                )
                                pkg_p = render(
                                    viewpoint,
                                    self.gaussians,
                                    self.pipeline_params,
                                    self.background,
                                    dynamic=False,
                                    dx=dxyz,
                                    ds=d_scale,
                                    dr=d_rot,
                                    mask=static_mask,
                                )
                                r_p, _ = tracking_residual_tensor(
                                    pkg_p, viewpoint, track_context
                                )

                                delta[j] = -float(step)
                                T_m = SE3_exp(delta) @ T_fd_base
                                viewpoint.update_RT(
                                    T_m[:3, :3], T_m[:3, 3]
                                )
                                pkg_m = render(
                                    viewpoint,
                                    self.gaussians,
                                    self.pipeline_params,
                                    self.background,
                                    dynamic=False,
                                    dx=dxyz,
                                    ds=d_scale,
                                    dr=d_rot,
                                    mask=static_mask,
                                )
                                r_m, _ = tracking_residual_tensor(
                                    pkg_m, viewpoint, track_context
                                )

                                dr_pm = r_p - r_m
                                J_audit[..., j] = dr_pm / (2.0 * step)
                                viewpoint.update_RT(
                                    R_fd_base, t_fd_base
                                )
                                vm = valid0.to(dtype=dr_pm.dtype)
                                denom = vm.sum().clamp_min(1.0)
                                diff_rms[j] = torch.sqrt(
                                    ((dr_pm * dr_pm) * vm).sum() / denom
                                )

                            viewpoint.update_RT(
                                R_fd_base, t_fd_base
                            )
                            viewpoint.cam_trans_delta.data.zero_()
                            viewpoint.cam_rot_delta.data.zero_()
                            diag_audit = tracking_jacobian_diagnostics(
                                residual0,
                                J_audit,
                                valid0,
                                cauchy_c=self.m2a2_cauchy_c,
                                damping=self.m2a2_damping,
                            )
                            family_rows.append(
                                {
                                    "eps": float(audit_eps),
                                    "j_rms": diag_audit[
                                        "j_rms"
                                    ].detach().cpu(),
                                    "diff_rms": diff_rms.detach().cpu(),
                                    "bread_eigenvalues": diag_audit[
                                        "bread_eigenvalues"
                                    ].detach().cpu(),
                                    "bread_condition": diag_audit[
                                        "bread_condition"
                                    ].detach().cpu(),
                                    "bread_trace": diag_audit[
                                        "bread_trace"
                                    ].detach().cpu(),
                                }
                            )
                        sweep[family] = family_rows
                    m2a2_fd_audit_payload = sweep

                m2a2_stats = cluster_robust_tracking_covariance(
                    residual0,
                    J_track,
                    valid0,
                    block_size=self.m2a2_block_size,
                    cauchy_c=self.m2a2_cauchy_c,
                    damping=self.m2a2_damping,
                    small_sample_correction=True,
                )
                m2a2_track_left = m2a2_stats["cov"]

                T_final_m2a2 = torch.eye(
                    4,
                    device=viewpoint.R.device,
                    dtype=viewpoint.R.dtype,
                )
                T_final_m2a2[:3, :3] = viewpoint.R
                T_final_m2a2[:3, 3] = viewpoint.T

                m2a2_track_right = left_covariance_to_right(
                    T_final_m2a2,
                    m2a2_track_left,
                )
                m2a2_post_right = fuse_prior_tracking_covariance(
                    viewpoint.pose_cov_abs_right,
                    m2a2_track_right,
                    information_scale=self.m2a2_information_scale,
                )

                # From this point onward, Camera.pose_cov_abs_right is the
                # posterior covariance used by the next frame's recursion.
                # The pre-update prior remains in local variables/diagnostics.
                viewpoint.pose_cov_abs_right = (
                    m2a2_post_right.detach().clone()
                )
                viewpoint.pose_cov_source = (
                    viewpoint.pose_cov_source + "+trackCR"
                )

        if self.m2a_enable and viewpoint.pose_cov_valid:
            with torch.no_grad():
                T_final = torch.eye(
                    4, device=viewpoint.R.device, dtype=viewpoint.R.dtype
                )
                T_final[:3, :3] = viewpoint.R
                T_final[:3, 3] = viewpoint.T

                if m2a_prior_pose is not None:
                    # Tracking pose optimization applies deterministic
                    # left-multiplicative re-centering. Right-invariant
                    # covariance is kept unchanged in M2-A shadow mode, but
                    # the correction magnitude is logged for the later
                    # tracking-information audit.
                    T_corr = T_final @ torch.linalg.inv(m2a_prior_pose)
                    corr_t = torch.linalg.norm(T_corr[:3, 3])
                    corr_trace = torch.clamp(
                        (torch.trace(T_corr[:3, :3]) - 1.0) / 2.0,
                        -1.0,
                        1.0,
                    )
                    corr_r = torch.acos(corr_trace)
                else:
                    corr_t = torch.tensor(
                        float("nan"),
                        device=T_final.device,
                        dtype=T_final.dtype,
                    )
                    corr_r = corr_t.clone()

                P_abs = viewpoint.pose_cov_abs_right
                P_raw = viewpoint.pose_cov_abs_right_raw
                P_prior_selected = (
                    m2a_abs_cal
                    if m2a_abs_cal is not None
                    else P_abs
                )
                eig_abs = torch.linalg.eigvalsh(
                    0.5 * (P_abs + P_abs.T)
                )
                sigma_t_abs = torch.sqrt(
                    torch.diagonal(P_abs)[:3].clamp_min(0.0)
                )
                sigma_r_abs = torch.sqrt(
                    torch.diagonal(P_abs)[3:].clamp_min(0.0)
                )

                if self.m2a_save_diagnostics:
                    m2_dir = os.path.join(
                        self.config["Results"]["save_dir"],
                        "m2a_pose_uncertainty",
                    )
                    os.makedirs(m2_dir, exist_ok=True)
                    torch.save(
                        {
                            "frame": int(viewpoint.uid),
                            "source": viewpoint.pose_cov_source,
                            "P_abs_right": P_abs.detach().cpu(),
                            "P_abs_prior_right": (
                                P_prior_selected.detach().cpu()
                            ),
                            "P_abs_right_raw": P_raw.detach().cpu(),
                            "P_track_left": (
                                m2a2_track_left.detach().cpu()
                                if m2a2_track_left is not None
                                else None
                            ),
                            "P_track_right": (
                                m2a2_track_right.detach().cpu()
                                if m2a2_track_right is not None
                                else None
                            ),
                            "P_abs_post_right": (
                                m2a2_post_right.detach().cpu()
                                if m2a2_post_right is not None
                                else None
                            ),
                            "P_rel_right": (
                                viewpoint.pose_cov_rel_right.detach().cpu()
                                if viewpoint.pose_cov_rel_right is not None
                                else None
                            ),
                            "T_final": T_final.detach().cpu(),
                            "T_gt": torch.cat(
                                [
                                    torch.cat(
                                        [
                                            viewpoint.R_gt.to(
                                                device=T_final.device,
                                                dtype=T_final.dtype,
                                            ),
                                            viewpoint.T_gt.to(
                                                device=T_final.device,
                                                dtype=T_final.dtype,
                                            ).reshape(3, 1),
                                        ],
                                        dim=1,
                                    ),
                                    torch.tensor(
                                        [[0.0, 0.0, 0.0, 1.0]],
                                        device=T_final.device,
                                        dtype=T_final.dtype,
                                    ),
                                ],
                                dim=0,
                            ).detach().cpu(),
                            "T_motion_prior": (
                                m2a_prior_pose.detach().cpu()
                                if m2a_prior_pose is not None
                                else None
                            ),
                            "tracking_correction_translation_m": (
                                corr_t.detach().cpu()
                            ),
                            "tracking_correction_rotation_rad": (
                                corr_r.detach().cpu()
                            ),
                            "sigma_t_abs_m": sigma_t_abs.detach().cpu(),
                            "sigma_r_abs_rad": sigma_r_abs.detach().cpu(),
                            "eig_abs": eig_abs.detach().cpu(),
                            "diag_calibration_enabled": bool(
                                self.m2a_use_diag_calibration
                                and self.m2a_diag_calibration is not None
                            ),
                            "diag_calibration_values": (
                                list(self.m2a_diag_calibration)
                                if (
                                    self.m2a_use_diag_calibration
                                    and self.m2a_diag_calibration is not None
                                )
                                else None
                            ),
                            "diag_calibration_source": (
                                self.m2a_diag_calibration_source
                            ),
                            "diag_calibration_train_sequences": list(
                                self.m2a_diag_calibration_train_sequences
                            ),
                            "m2a2_enabled": bool(self.m2a2_enable),
                            "m2a2_block_size": int(self.m2a2_block_size),
                            "m2a2_information_scale": float(
                                self.m2a2_information_scale
                            ),
                            "m2a2_cluster_count": (
                                int(m2a2_stats["cluster_count"])
                                if m2a2_stats is not None
                                else 0
                            ),
                            "m2a2_num_pixels": (
                                int(m2a2_stats["num_pixels"])
                                if m2a2_stats is not None
                                else 0
                            ),
                            "m2a2_num_observations": (
                                int(m2a2_stats["num_observations"])
                                if m2a2_stats is not None
                                else 0
                            ),
                            "m2a2_condition": (
                                m2a2_stats["condition"].detach().cpu()
                                if m2a2_stats is not None
                                else None
                            ),
                            "m2a2_j_rms": (
                                jac_diag["j_rms"].detach().cpu()
                                if m2a2_stats is not None
                                else None
                            ),
                            "m2a2_bread_eigenvalues": (
                                jac_diag[
                                    "bread_eigenvalues"
                                ].detach().cpu()
                                if m2a2_stats is not None
                                else None
                            ),
                            "m2a2_bread_condition": (
                                jac_diag[
                                    "bread_condition"
                                ].detach().cpu()
                                if m2a2_stats is not None
                                else None
                            ),
                            "m2a2_bread_trace": (
                                jac_diag["bread_trace"].detach().cpu()
                                if m2a2_stats is not None
                                else None
                            ),
                            "m2a2_fd_audit": m2a2_fd_audit_payload,
                            "m2a2_rgb_scale": (
                                track_context["rgb_scale"].detach().cpu()
                                if self.m2a2_enable
                                and m2a2_stats is not None
                                else None
                            ),
                            "m2a2_depth_scale": (
                                track_context["depth_scale"].detach().cpu()
                                if self.m2a2_enable
                                and m2a2_stats is not None
                                else None
                            ),
                        },
                        os.path.join(
                            m2_dir, f"{int(viewpoint.uid):06d}.pt"
                        ),
                    )

                Log(
                    "M2-A frame", viewpoint.uid,
                    "source", viewpoint.pose_cov_source,
                    "calib_source", self.m2a_diag_calibration_source,
                    "sigma_t_abs", sigma_t_abs.detach().cpu().tolist(),
                    "sigma_r_abs", sigma_r_abs.detach().cpu().tolist(),
                    "eig_min", float(eig_abs.min().detach().cpu()),
                    "track_dt", float(corr_t.detach().cpu()),
                    "track_dr", float(corr_r.detach().cpu()),
                    "m2a2_clusters", (
                        int(m2a2_stats["cluster_count"])
                        if m2a2_stats is not None else 0
                    ),
                    "m2a2_obs", (
                        int(m2a2_stats["num_observations"])
                        if m2a2_stats is not None else 0
                    ),
                    "m2a2_jrms", (
                        jac_diag["j_rms"].detach().cpu().tolist()
                        if m2a2_stats is not None else None
                    ),
                    "m2a2_bread_cond", (
                        float(
                            jac_diag[
                                "bread_condition"
                            ].detach().cpu()
                        )
                        if m2a2_stats is not None else float("nan")
                    ),
                    tag="Frontend",
                )

        
        
        with torch.no_grad():
            render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot)
            
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))
            
            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap, add_new_gaussian=True, dynamic_render=False):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, add_new_gaussian, dynamic_render]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 1 == 0:
            torch.cuda.empty_cache()
    
            
    def run(self):
        # init
        cur_frame_idx = 0
        last_keyframe_idx = 0
        # projection_matrix for viewpoints
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        # 
        dynamic_render=False
        keyframe_list = [0]
        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        0,
                        final=True,
                        monocular=self.monocular,
                    )
                    if self.save_results:
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                if self.requested_keyframe > 0 and (cur_frame_idx - last_keyframe_idx) >= self.kf_interval:
                    time.sleep(0.01)
                    continue
                    
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )
                
                # Tracking
                start_time = time.time()
                render_pkg = self.tracking(cur_frame_idx, viewpoint, last_keyframe_idx)
                tracking_time = time.time() - start_time
                print('Tracking time: ', tracking_time)
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                if self.dynamic_model and self.gaussians.init_deform == 'mlp':
                    self.gaussians.deform.deform.reg_loss = 0.
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx-1)
                    cur_frame_idx += 1
                    print("skip")
                    continue

                last_keyframe_idx = self.current_window[0]
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                
                forced_by_gap5 = ((cur_frame_idx - last_keyframe_idx) >= 5)
                forced_by_dystart = (cur_frame_idx == self.dystart)
                geometric_kf = bool(create_kf)
                create_kf = forced_by_gap5 or create_kf or forced_by_dystart

                forced_by_new_object = (
                    self.dataset.dynamic_objects > self.dynamic_objects
                    and cur_frame_idx > 0
                )
                if forced_by_new_object:
                    create_kf = True
                    new_object = True
                else:
                    new_object = False

                if self.m1_keyframe_reason_log and create_kf:
                    Log(
                        "KF reason",
                        "frame", int(cur_frame_idx),
                        "last", int(last_keyframe_idx),
                        "gap", int(cur_frame_idx - last_keyframe_idx),
                        "check_time", bool(check_time),
                        "geometry", bool(geometric_kf),
                        "gap5", bool(forced_by_gap5),
                        "dystart", bool(forced_by_dystart),
                        "new_object", bool(forced_by_new_object),
                        tag="Frontend",
                    )
                    
                if create_kf:
                    keyframe_list.append(cur_frame_idx)
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map, True, dynamic_render
                    )
                    
                    temp_log = ("create keyframe:", cur_frame_idx, 
                                "add new gaussian:", True, 
                                "point_ratio:", point_ratio, 
                                "dynamic_render:", dynamic_render)
                    
                    Log(tag="Frontend", *temp_log)
                    self.cameras[cur_frame_idx-1].clean_key()
                else:
                    self.cleanup(cur_frame_idx-1)
                cur_frame_idx += 1
                self.dynamic_objects = self.dataset.dynamic_objects
                if (
                    # self.save_results
                    # and self.save_trj
                    self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
