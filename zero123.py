# using the init as a template, generate colmap formatted data
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from utils.pose_utils import *
import json


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
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


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws


def get_circular_camera_poses(M=120, radius=2.5, elevation=30.0):
    # M: number of circular views
    # radius: camera dist to center
    # elevation: elevation degrees of the camera
    # return: (M, 4, 4)
    assert M > 0 and radius > 0

    elevation = np.deg2rad(elevation)

    camera_positions = []
    for i in range(M):
        azimuth = 2 * np.pi * i / M
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = torch.from_numpy(camera_positions).float()
    extrinsics = center_looking_at_camera_pose(camera_positions)
    return extrinsics


def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics


def get_zero123plus_input_cameras(batch_size=1, radius=4.0, fov=30.0):
    """
    Get the input camera parameters.
    """
    azimuths = np.array([30, 90, 150, 210, 270, 330, 20]).astype(float)
    elevations = np.array([20, -10, 20, -10, 20, -10, 20]).astype(float)
    
    c2ws = spherical_camera_pose(azimuths, elevations, radius)
    c2ws = c2ws.float().flatten(-2)

    Ks = FOV_to_intrinsics(fov).unsqueeze(0).repeat(6, 1, 1).float().flatten(-2)

    extrinsics = c2ws[:, :12]
    intrinsics = torch.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    cameras = torch.cat([extrinsics, intrinsics], dim=-1)

    return cameras.unsqueeze(0).repeat(batch_size, 1, 1)


def tocolmap(
    root_dir='data/zero123-2/sparse/0',
    input_view_num=6,
    input_image_size=320,
    fov=30,
):
    root_dir = Path(root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    paths = sorted(os.listdir(root_dir))
    print('============= length of dataset %d =============' % len(paths))

    cam_distance = 4.0
    azimuths = np.array([30, 90, 150, 210, 270, 330, 0])
    elevations = np.array([20, -10, 20, -10, 20, -10, 0])
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    x = cam_distance * np.cos(elevations) * np.cos(azimuths)
    y = cam_distance * np.cos(elevations) * np.sin(azimuths)
    z = cam_distance * np.sin(elevations)

    cam_locations = np.stack([x, y, z], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()
    c2ws = center_looking_at_camera_pose(cam_locations)
    c2ws = c2ws.float()
    K = FOV_to_intrinsics(fov).float()
    print(c2ws, c2ws.shape, K)


    
    # save c2w as colmap format camera.txt, images named {1~6}.png
    with open(root_dir / 'images.txt', 'w') as f:
        prem = \
'''# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 6, mean observations per image: 0'''
        f.write(prem)
        for i in range(7):
            c2w = c2ws[i]
            c2w[:3, 1:3] *= -1
            w2c = torch.inverse(c2w)
            quaternion = torch.from_numpy(rotmat2qvec(w2c[:3, :3].numpy()))
            qt = torch.cat([quaternion.unsqueeze(0), w2c[:3, 3].unsqueeze(0)], dim=-1)
            qt = qt.flatten().tolist()
            qt = [str(x) for x in qt]
            str_line = f'\n{i} ' + ' '.join(qt) + f' 1 {i+1}.png\n'
            f.write(str_line)
    
    intr = torch.tensor([K[0,0], K[0,2]]) * input_image_size
    focal, center = intr.tolist()
    with open(root_dir / 'cameras.txt', 'w') as f:
        prem = \
'''# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
'''
        f.write(prem)
        f.write(f'1 PINHOLE {input_image_size} {input_image_size} {focal} {focal} {center} {center}\n')


def toblender(
    root_dir='data/0123/',
    input_view_num=6,
    input_image_size=320,
    fov=30,
):
    root_dir = Path(root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    paths = sorted(os.listdir(root_dir))
    print('============= length of dataset %d =============' % len(paths))

    cam_distance = 4.0
    azimuths = np.array([30, 90, 150, 210, 270, 330])
    elevations = np.array([20, -10, 20, -10, 20, -10])
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    x = cam_distance * np.cos(elevations) * np.cos(azimuths)
    y = cam_distance * np.cos(elevations) * np.sin(azimuths)
    z = cam_distance * np.sin(elevations)

    cam_locations = np.stack([x, y, z], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()
    c2ws = center_looking_at_camera_pose(cam_locations)
    c2ws = c2ws.float()
    K = FOV_to_intrinsics(fov).float()
    intr = torch.tensor([K[0,0], K[0,2]]) * input_image_size
    focal, center = intr.tolist()

    frames = []
    for i in range(6):
        cam = {}
        cam['file_path'] = f'./train/{i+1}'
        cam['rotation'] = 0.012
        c2w = c2ws[i]
        # c2w[:3, 1:3] *= -1
        # w2c = torch.inverse(c2w)
        cam['transform_matrix'] = c2w.tolist()
        frames.append(cam)

    js = {
        'camera_angle_x': np.deg2rad(fov),
        'frames': frames
    }
    with open(root_dir / 'transforms_train.json', 'w', encoding='utf-8') as f:
        json.dump(js, f, indent=4)

    

if __name__ == '__main__':
    tocolmap()
    # toblender()
    # w2c = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # print(rotmat2qvec(w2c))

