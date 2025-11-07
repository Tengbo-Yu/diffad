import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MotionAlignment(nn.Module):
    def __init__(self, pc_range, grid_reso, mode='bilinear'):
        super().__init__()

        self.mode = mode
        ego_to_image = gen_ego_to_image(pc_range, grid_reso)
        image_to_ego = torch.inverse(ego_to_image)
        self.register_buffer("ego_to_image", ego_to_image)
        self.register_buffer("image_to_ego", image_to_ego)

    def forward(self, x, ego0_to_world, ego1_to_world):
        world_to_ego0 = torch.inverse(ego0_to_world)
        ego1_to_ego0 = world_to_ego0 @ ego1_to_world
        ego1_to_ego0 = ego1_to_ego0[:, [0, 1, 3], :][:, :, [0, 1, 3]]
        ego0_to_ego1 = torch.inverse(ego1_to_ego0)
        trans_matrix = self.ego_to_image @ ego0_to_ego1 @ self.image_to_ego
        trans_matrix = trans_matrix[:, :2, :]
        grid = F.affine_grid(trans_matrix, size=x.shape, align_corners=False)
        warped_x = F.grid_sample(x, grid.float(), mode=self.mode, padding_mode='zeros',
                                 align_corners=False)
        return warped_x


def gen_ego_to_image(pc_range, grid_reso):
    """BEV2Image
        y(front)
        ↑
        |
        |
        0 -----> x(right)
    bev: x --> right, y --> front
    img: u --> right, v --> back
    u = (x_min + x) / r_x,
    v = (y_min - y) / r_y
    """
    x_min, y_min, _, x_max, y_max, _ = pc_range
    r_x, r_y = grid_reso[:2]

    H, W = int((y_max - y_min) / r_y), int((x_max - x_min) / r_x)

    ego_to_pixel = np.array(
        [[1 / r_x, 0.0, x_min / r_x],
         [0.0, -1.0 / r_y, y_min / r_y],
         [0.0, 0.0, 1.0]]
    )

    pixel_to_image = np.array(
        [[2.0 / W, 0.0, 1.0 / W - 1],
         [0.0, 2.0 / H, 1.0 / H - 1],
         [0.0, 0.0, 1.0]]
    )
    ego_to_image = pixel_to_image @ ego_to_pixel
    ego_to_image = ego_to_image.astype(np.float32)
    return torch.from_numpy(ego_to_image)
