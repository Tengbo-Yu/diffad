import numpy as np
import cv2
from mmcv.image import imread
from util.nms_util import Box3d
from util.color_util import ColorBoard, interpolate_color
from util.projection_util import proj_traj_to_pv_wide_path, draw_faded_path


class BevVisualizer(object):
    def __init__(self, config):
        self.global_config = config['Global']
        self.pc_range = self.global_config['point_cloud_range']
        self.cam_names = ['cam_front', 'cam_front_right',
                          'cam_front_left', 'cam_back',
                          'cam_back_left', 'cam_back_right']

    def visualize(self, filenames, lidar2img, pred_image, gt_image, cur_gt_plan_traj_vcs, cur_pred_plan_traj_vcs,
                  cur_command):
        img_h, img_w = 540, 960
        image = {}
        for cam_id, filename in enumerate(filenames):
            cur_cam_img = imread(filename, channel_order='rgb')
            cam_name = self.cam_names[cam_id]
            cur_cam_img = cv2.resize(cur_cam_img, (img_w, img_h))
            _cam_name = filename.split('/')[-2]  # rgb_xxx
            if 'rgb' in _cam_name:  # b2d data
                cam_name = cam_name.replace('left', 'right') if 'left' in cam_name \
                    else cam_name.replace('right', 'left')

            # Draw camera name on the top left of the image
            cv2.putText(cur_cam_img, cam_name.replace('cam_', ''), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            image[cam_name] = cur_cam_img

        # Draw command on the top middle of the image
        # https://github.com/Thinklab-SJTU/Bench2Drive/issues/9
        cur_command = np.argmax(cur_command)
        command_text = ['Left', 'Right', 'Straight', 'LaneFollow', 'ChangeLaneLeft', 'ChangeLaneRight'][cur_command]
        cv2.putText(image['cam_front'], command_text, (img_w // 2 - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        ## 
        total_img_h = img_h * 2
        total_img_w = img_w * 3
        image_canvas = np.zeros(shape=(total_img_h, total_img_w, 3), dtype=np.uint8)
        image_canvas[:img_h, :img_w, :] = image['cam_front_left']
        image_canvas[:img_h, img_w:img_w * 2, :] = image['cam_front']
        image_canvas[:img_h, img_w * 2:, :] = image['cam_front_right']
        image_canvas[img_h:img_h * 2, :img_w, :] = image['cam_back_left']
        image_canvas[img_h:img_h * 2, img_w:img_w * 2, :] = image['cam_back']
        image_canvas[img_h:img_h * 2, img_w * 2:, :] = image['cam_back_right']

        bev_resize_h = total_img_h
        bev_h, bev_w = gt_image.shape[1:]  # c,h,w*3
        bev_resize_w = int(bev_resize_h / bev_h * bev_w)

        gt_image = np.clip((gt_image + 1) * 128, 0, 255)
        gt_image = np.transpose(gt_image, (1, 2, 0)).astype(np.uint8)
        gt_image = resize_and_pad_image(gt_image, (bev_resize_h, bev_resize_w))

        pred_image = np.clip((pred_image + 1) * 128, 0, 255)
        pred_image = np.transpose(pred_image, (1, 2, 0)).astype(np.uint8)
        pred_image = resize_and_pad_image(pred_image, (bev_resize_h, bev_resize_w))

        gt_bevmap_image, gt_motion_image, gt_planning_image = np.split(gt_image, 3, axis=1)
        pred_bevmap_image, pred_motion_image, pred_planning_image = np.split(pred_image, 3, axis=1)

        gt_wrap_image = gt_bevmap_image
        pred_wrap_image = pred_bevmap_image

        ### draw ego
        bev_reso = self.global_config['bev_reso']
        x_min, y_min, _, x_max, y_max, _ = self.pc_range

        bev_old_h, bev_old_w = (int((y_max - y_min) / bev_reso[0]), int((x_max - x_min) / bev_reso[0]))
        bev_new_h, bev_new_w = pred_wrap_image.shape[:2]
        grid_ratio = (bev_new_h / bev_old_h, bev_new_w / bev_old_w)
        draw_ego_car(gt_wrap_image, x_max, y_max, bev_reso, grid_ratio)
        draw_ego_car(pred_wrap_image, x_max, y_max, bev_reso, grid_ratio)

        ### wrap motion
        mask = (gt_motion_image != 0).astype(np.uint8)
        gt_wrap_image = gt_motion_image * mask + gt_wrap_image * (1 - mask)

        threshold = 10
        front_mask = np.mean(pred_motion_image, axis=-1, keepdims=True)
        front_mask = (front_mask >= threshold).astype(np.uint8)
        pred_wrap_image = pred_motion_image * front_mask + pred_wrap_image * (1 - front_mask)
        ### wrap motion

        ### wrap planning
        mask = (gt_planning_image != 0).astype(np.uint8)
        gt_wrap_image = gt_planning_image * mask + gt_wrap_image * (1 - mask)

        threshold = 10
        front_mask = np.mean(pred_planning_image, axis=-1, keepdims=True)
        front_mask = (front_mask >= threshold).astype(np.uint8)
        pred_wrap_image = pred_planning_image * front_mask + pred_wrap_image * (1 - front_mask)

        # draw planning onto front view
        if 'cam_front':
            canvas = image_canvas[:img_h, img_w:img_w * 2, :]
        else:
            canvas = image_canvas[img_h:img_h * 2, img_w:img_w * 2, :]

        ori_img_h, ori_img_w = self.global_config['input_size']['cam_front']
        # draw gt planning
        cam_pts_3d = proj_traj_to_pv_wide_path(cur_gt_plan_traj_vcs, lidar2img, ori_img_h, ori_img_w, traj_width=2.)
        if len(cam_pts_3d) != 0:
            # recale
            cam_pts_3d[..., 0] = cam_pts_3d[..., 0] / ori_img_w * img_w
            cam_pts_3d[..., 1] = cam_pts_3d[..., 1] / ori_img_h * img_h
            cam_pts_2d = cam_pts_3d[..., :2].astype(np.int32)

            # draw trajectory in faded away
            draw_faded_path(cam_pts_2d, canvas, ColorBoard.blue, min_alpha=0.)

        # draw model planning
        cam_pts_3d = proj_traj_to_pv_wide_path(cur_pred_plan_traj_vcs, lidar2img, ori_img_h, ori_img_w, traj_width=2.)
        if len(cam_pts_3d) != 0:
            # recale
            cam_pts_3d[..., 0] = cam_pts_3d[..., 0] / ori_img_w * img_w
            cam_pts_3d[..., 1] = cam_pts_3d[..., 1] / ori_img_h * img_h
            cam_pts_2d = cam_pts_3d[..., :2].astype(np.int32)

            # draw trajectory in faded away
            draw_faded_path(cam_pts_2d, canvas, ColorBoard.red, min_alpha=0.)

            # draw planning on bev
            draw_ego_planning(cur_pred_plan_traj_vcs, pred_wrap_image, x_max, y_max, bev_reso, grid_ratio)

        ### wrap planning

        all_canvas = []
        all_canvas.append(gt_wrap_image)
        all_canvas.append(image_canvas)
        all_canvas.append(pred_wrap_image)
        all_canvas = cv2.hconcat(all_canvas)
        all_canvas = cv2.resize(all_canvas, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        return np.transpose(all_canvas, (2, 0, 1))


def resize_and_pad_image(image, canvas_size):
    """
    Resize and pad an image to fit the specified canvas size.
    
    :param image: Input image as a numpy array.
    :param canvas_size: Tuple of (H, W) specifying the size of the canvas.
    :return: Resized and padded image as a numpy array.
    """
    h, w = image.shape[:2]
    H, W = canvas_size

    # Calculate the scaling factor
    scale = min(W / w, H / h)

    # Compute new image dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # # Calculate padding to center the image
    # pad_top = (H - new_h) // 2
    # pad_bottom = H - new_h - pad_top
    # pad_left = (W - new_w) // 2
    # pad_right = W - new_w - pad_left

    # # Pad the resized image
    # padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return resized_image


def draw_ego_car(bev_canvas, x_max, y_max, bev_reso, bev_ratio):
    ego_l = 4.5  # fake data
    ego_w = 1.8  # fake data
    cam2vcs_distance = 1.0  # fake data
    yaw = 0
    ego_box = Box3d(ego_l / 2 - cam2vcs_distance, 0, 0, ego_l, ego_w, 0, yaw)
    pts = ego_box.bottom_corners()[:2].T
    # reverse x and swap x and y
    coords_x = (x_max + pts[:, 1]) / bev_reso[1]
    coords_y = (y_max - pts[:, 0]) / bev_reso[0]
    coords = np.stack([coords_x, coords_y], axis=-1)
    coords = (coords * bev_ratio).astype(int)

    ego_color = ColorBoard.gray  # silvery gray
    ego_points = np.array(
        [tuple(coords[0][:2]), tuple(coords[1][:2]),
         tuple(coords[2][:2]), tuple(coords[3][:2])])
    cv2.fillPoly(bev_canvas, [ego_points], ego_color)


def draw_ego_planning(pts, bev_canvas, x_max, y_max, bev_reso, bev_ratio):
    coords_y = (y_max - pts[:, 1]) / bev_reso[1]
    coords_x = (x_max + pts[:, 0]) / bev_reso[0]
    coords = np.stack([coords_x, coords_y], axis=-1)
    coords = (coords * bev_ratio).astype(int)
    for i, point in enumerate(coords):
        t = i / (len(coords) - 1)
        color = interpolate_color(ColorBoard.red, ColorBoard.yellow, t)
        cv2.circle(bev_canvas, point, radius=6, color=color, thickness=-1)
