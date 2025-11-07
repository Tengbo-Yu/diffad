import numpy as np
import cv2
import torch
from shapely.geometry import LineString


def proj_traj_to_pv_wide_path(pt_2d, lidar2img, img_h=768, img_w=1920, traj_width=1., camera_name='cam_front',
                              num_pts=100):
    """
        pt_2d: array(N, 2), vcs trajectory
    """
    # smoothing trajectory path by interpolation
    pts = LineString(pt_2d)
    distances = np.linspace(0, pts.length, num_pts)
    sampled_pts = np.array([list(pts.interpolate(distance).coords) for distance in distances])
    sampled_pts = sampled_pts.reshape(-1, 2)

    expand_points = []
    for i in range(len(sampled_pts) - 1):
        point = sampled_pts[i]
        next_point = sampled_pts[i + 1]

        # vertical direction
        direction = next_point - point
        normal_vector = np.array([-direction[1], direction[0]])
        normal = normal_vector / (np.linalg.norm(normal_vector) + 1e-9)

        # lateral expanded points
        offset = normal * traj_width / 2.
        expand_points.append([point - offset, point + offset])

    # end point
    expand_points.append([next_point - offset, next_point + offset])
    expand_points = np.concatenate(expand_points, axis=0)  # (N*2, 2)

    # proj onto image
    pt_zs = np.ones((expand_points.shape[0], 1), dtype=pt_2d.dtype) * -2.
    pt_ones = np.ones((expand_points.shape[0], 1), dtype=pt_2d.dtype)
    pt_4d = np.concatenate([expand_points, pt_zs, pt_ones], axis=-1)

    lidar2img = lidar2img[camera_name]
    pts_2d = pt_4d @ lidar2img.T
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    fov_inds = ((pts_2d[:, 0] < img_w)
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img_h)
                & (pts_2d[:, 1] >= 0))

    imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d
    return imgfov_pts_2d


def draw_faded_path(pt_2d, canvas, color=(0, 0, 255), min_alpha=0.):
    """
        pt_2d: array([T*2, 2]), lateral expanded path
    """
    H, W, C = canvas.shape
    boundry_line = H - 0
    T = len(pt_2d) // 2

    # set boundry point as the first point for better visualization
    first_point = pt_2d[:2]
    if max(first_point[:, 1]) < boundry_line:
        first_point[:, 1] = boundry_line
        pt_2d = np.concatenate([first_point, pt_2d], axis=0)

    for i in range(T):
        alpha = max(1. - (i / (T - 1 + 1e-9)), min_alpha)
        alpha_255 = int(255 * alpha)
        color_with_alpha = (*color, alpha_255)
        overlay = canvas.copy()

        point = pt_2d[i * 2:(i + 1) * 2]
        next_i = min(T - 1, i + 1)
        next_point = pt_2d[next_i * 2:(next_i + 1) * 2]

        # clip near-boundry points
        if min(point[:, 1]) > boundry_line and max(next_point[:, 1]) < boundry_line:
            point[:, 1] = boundry_line

        # start drawing from boundry
        if min(point[:, 1]) > boundry_line:
            continue
        pts = np.array([point[0], point[1], next_point[1], next_point[0]], dtype=np.int32).reshape(-1,
                                                                                                   2)  # counter-clockwise
        cv2.fillPoly(overlay, [pts], color=color_with_alpha)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def img2lidar(x, pc_range):
    img_pts = x[..., :2]
    others = x[..., 2:]
    vcs_x = img_pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    vcs_y = - (img_pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    if isinstance(x, torch.Tensor):
        result = torch.cat([vcs_x, vcs_y, others], dim=-1)
    elif isinstance(x, np.ndarray):
        result = np.concatenate([vcs_x, vcs_y, others], axis=-1)
    else:
        raise ValueError(f'{type(x)} is not supported')
    return result
