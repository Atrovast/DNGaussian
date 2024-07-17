#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
from os import makedirs
from random import randint
from utils.graphics_utils import fov2focal
from utils.loss_utils import l1_loss, patch_norm_mse_loss, patch_norm_mse_loss_global, ssim
# from utils.loss_utils import mssim as ssim
from gaussian_renderer import render, render_for_depth, render_for_opa   # , network_gui
from gaussian_renderer import render_sh, render_for_depth_sh   # , network_gui

import sys
from scene import Scene, GaussianModel,  GaussianModelSH

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    print('Launch TensorBoard')
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import cv2
import glob
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def training_sh(dataset, opt, pipe, testing_iterations, saving_iterations):
    first_iter = 0
    gaussians = GaussianModelSH(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ascii=True, dynamic_ncols=True)
    first_iter += 1

    patch_range = (5, 17) # LLFF

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(max(iteration - opt.position_lr_start, 0))

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        # bg_mask = None
        bg_mask = (gt_image.min(0, keepdim=True).values > 254/255)

        # -------------------------------------------------- DEPTH --------------------------------------------
        if iteration > opt.hard_depth_start and iteration < opt.densify_until_iter and iteration % 10 == 0:
            render_pkg = render_for_depth_sh(viewpoint_cam, gaussians, pipe, background)
            depth = render_pkg["depth"]

            # Depth loss
            loss_hard = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono
            depth_mono[bg_mask] = 0


            loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_hard += 0.1 * loss_l2_dpt


            loss_global = patch_norm_mse_loss_global(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_hard += 1 * loss_global

            loss_hard.backward()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


            # if iteration > opt.densify_from_iter:
            #     gaussians.prune(opt.prune_threshold)


        # ---------------------------------------------- Photometric --------------------------------------------
        
        render_pkg = render_sh(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # depth
        depth, opacity, alpha = render_pkg["depth"], render_pkg["opacity"], render_pkg['alpha']  # [visibility_filter]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


        loss.backward()
        
        # ================================================================================

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not loss.isnan():
                ema_loss_for_log = 0.4 * (loss.item()) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            clean_iterations = testing_iterations + [first_iter]
            clean_views(iteration, clean_iterations, scene, gaussians, pipe, background)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                ply_path = scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter and iteration not in clean_iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if 'chair' not in scene.source_path: 
                        color = render_sh(viewpoint_cam, gaussians, pipe, background)["color"]
                        white_mask = color.min(-1, keepdim=True).values > 253/255
                        gaussians.xyz_gradient_accum[white_mask] = 0
                        # gaussians._opacity[white_mask] = gaussians.inverse_opacity_activation(torch.ones_like(gaussians._opacity[white_mask]) * 0.1)
                        gaussians._opacity[white_mask] = gaussians.inverse_opacity_activation(gaussians.opacity_activation(gaussians._opacity[white_mask]) * 0.1)
                    
                    if 'ship' in scene.source_path: 
                        gaussians.prune_points(gaussians.get_xyz[:,-1] < -0.5)  
                    if 'hotdog' in scene.source_path: 
                        gaussians.prune_points(gaussians.get_xyz[:,-1] < -0.2)       
                

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
    
    
    # render video
    from scene.cameras import Camera
    with torch.no_grad():
        cam_distance = 4.0
        azimuths = np.array(range(0, 360, 3))
        elevations = np.array([20 for _ in range(0, 360, 3)])
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)

        x = cam_distance * np.cos(elevations) * np.cos(azimuths)
        y = cam_distance * np.cos(elevations) * np.sin(azimuths)
        z = cam_distance * np.sin(elevations)

        cam_locations = np.stack([x, y, z], axis=-1)
        cam_locations = torch.from_numpy(cam_locations).float()
        c2ws = center_looking_at_camera_pose(cam_locations)
        c2ws = c2ws.float()
        c2ws[:, :3, 1:3] *= -1
        images = []
        for i in range(0, 360 // 3):
            c2w = c2ws[i]
            w2c = np.linalg.inv(c2w.numpy())
            R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]
            cam = Camera(i, R, T, np.deg2rad(30), np.deg2rad(30), None, None, i, i, None)
            cam.image_width = 320
            cam.image_height = 320
            render_pkg = render_sh(cam, gaussians, pipe, background)
            image = render_pkg["render"]
            images.append(image)
    vid = torch.stack(images)
    
    return ply_path, torch.stack(images)

def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    import torch.nn.functional as F
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics

@torch.no_grad()
def clean_views(iteration, test_iterations, scene, gaussians, pipe, background):
    if iteration in test_iterations:
        visible_pnts = None
        for viewpoint_cam in scene.getTrainCameras().copy():
            render_pkg = render_sh(viewpoint_cam, gaussians, pipe, background)
            visibility_filter = render_pkg["visibility_filter"]
            if visible_pnts is None:
                visible_pnts = visibility_filter
            visible_pnts += visibility_filter
        unvisible_pnts = ~visible_pnts
        gaussians.prune_points(unvisible_pnts)


def predict_depth(img_path):
    root_path_1 = img_path+'/*png'
    image_paths = sorted(glob.glob(root_path_1))
    output_path = os.path.join('/'.join(root_path_1.split('/')[:-1]), 'depth_maps')
    if not os.path.exists(output_path): 
        os.makedirs(output_path, exist_ok=True)
    else:
        return

    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    print('image_paths:', image_paths)

    # for image_paths, output_path in zip(image_path_pkg, output_path_pkg):
    for k in range(len(image_paths)):
        filename = image_paths[k]
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8), interpolation=cv2.INTER_CUBIC)
        # print('k, img.shape:', k, img.shape) #(1213, 1546, 3)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch).squeeze()

        output = prediction.cpu().numpy()
        name = 'depth_'+filename.split('/')[-1]
        print('######### output_path and name:', output_path,  name)
        output_file_name = os.path.join(output_path, name.split('.')[0])
        write_depth(output_file_name, output, bits=2)


def write_depth(path, depth, bits=1, absolute_depth=False):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    if absolute_depth:
        out = depth
    else:
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2 ** (8 * bits)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
    # print('depth:', depth.min(), depth.max())
    # print('out:', out.min(), out.max())
    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return

def train(img_path):
    predict_depth(img_path)

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    args.source_path = img_path
    args.model_path = img_path
    args.rand_pcd = True
    # safe_state(True)
    ply_path, rendered_images = training_sh(lp.extract(args), op.extract(args), pp.extract(args), [3000, 6000], [3000, 6000])
    print("\nTraining complete.")
    return ply_path, rendered_images
