import cv2
import numpy as np
from util.nms_util import Box3d
from util.color_util import ColorBoard, interpolate_color

class2color = {
    'car': ColorBoard.white,
    'van': ColorBoard.white,
    'bus': ColorBoard.blue,
    'truck': ColorBoard.green,
    "construction_vehicle": ColorBoard.green,
    "bicycle": ColorBoard.purple,
    "pedestrian": ColorBoard.cyan,
    "motorcycle": ColorBoard.yellow,
    "trailer": ColorBoard.purple,
    "traffic_cone": ColorBoard.red,
    "traffic_light": ColorBoard.pink,
    "traffic_sign": ColorBoard.pink,
    "barrier": ColorBoard.red,
    "others": ColorBoard.pink
}

COLOR_MAP = {
    'red': ColorBoard.red,
    'white': ColorBoard.white,
    'yellow': ColorBoard.yellow,
    'purple': ColorBoard.purple,
    'teal': ColorBoard.teal,
    'gray': ColorBoard.gray,
    'blue': ColorBoard.blue,
    'orange': ColorBoard.orange,
    'unknown': ColorBoard.gray,
    'other': ColorBoard.gray
}


class Rasterize(object):
    "rasterize label onto a BEV map, where ego heading to front"

    def __init__(self, box_type='lidar') -> None:
        # TODO: pc_range, bev_reso
        box_type_lower = box_type.lower()
        if box_type_lower == 'lidar':
            # x-->right, y-->front
            pass

    @staticmethod
    def draw_polyline(coords, label, canvas):
        label = label.lower()
        if label == 'contours':
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.red, thickness=3)
        elif label == 'road_divider':
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.white, thickness=3)
        elif label == 'lane_divider':
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.white, thickness=3)
        elif label == 'ped_crossing':
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.teal, thickness=3)
        elif label == 'others':
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.purple, thickness=3)
        elif label == 'solid':
            # solid lane
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.white, thickness=3)
        elif label == 'broken':
            # dash lane
            for i in range(0, len(coords) - 1, 2):
                cv2.line(canvas, tuple(coords[i]), tuple(coords[i + 1]), color=ColorBoard.white, thickness=3)
        elif label == 'solidsolid':
            # double solid lane
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.yellow, thickness=3)
        elif label == 'center':
            # centerline
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.green, thickness=2)
        elif label == 'trafficlight':
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.orange, thickness=2)
        elif label == 'stopsign':
            cv2.polylines(canvas, np.int32([coords]), False, color=ColorBoard.pink, thickness=4)
        else:
            raise NameError('Unsupported type {}!'.format(label))

    @staticmethod
    def draw_bev_static_map(instances, instances_labels, x_max, y_max, bev_reso, canvas):
        for pts, label in zip(instances, instances_labels):
            if len(pts) < 2:
                continue
            coords_y = (y_max - pts[:, 1]) / bev_reso[1]
            coords_x = (x_max + pts[:, 0]) / bev_reso[0]
            coords = np.stack([coords_x, coords_y], axis=-1).astype(np.int32)
            assert len(coords) >= 2
            Rasterize.draw_polyline(coords, label, canvas)

        return canvas

    @staticmethod
    def draw_each_bbox(pts, category, canvas, x_max, y_max, bev_reso):
        coords_y = (y_max - pts[:, 1]) / bev_reso[1]
        coords_x = (x_max + pts[:, 0]) / bev_reso[0]
        coords = np.stack([coords_x, coords_y], axis=-1).astype(np.int32)
        assert len(coords) >= 2

        color = class2color[category]
        ego_points = np.array(
            [tuple(coords[0][:2]), tuple(coords[1][:2]),
             tuple(coords[2][:2]), tuple(coords[3][:2])])
        cv2.fillPoly(canvas, [ego_points], color)

    @staticmethod
    def draw_bev_obstacle_map(bboxes_3d, labels_3d, x_max, y_max, bev_reso, canvas):
        """
            lidar-box3d
                    .. code-block:: none

                                up z    x front (yaw=-0.5*pi)
                                    ^   ^
                                    |  /
                                    | /
            (yaw=-pi) left y <------ 0 -------- (yaw=0)
        """
        for bbox_tensor, bbox_label in zip(bboxes_3d, labels_3d):
            category = bbox_label
            bbox = bbox_tensor.numpy()
            X, Y, Z = bbox[:3]
            L, W, H = bbox[3:6]
            yaw = bbox[6]  # yaw=0 --> right
            box_yaw = -yaw  # TODO: figure out coordinate
            yaw = np.remainder(yaw, np.pi * 2)
            box = Box3d(X, Y, Z, L, W, H, box_yaw)
            pts = box.bottom_corners()[:2].T
            Rasterize.draw_each_bbox(pts, category, canvas, x_max, y_max, bev_reso)
        return canvas

    @staticmethod
    def draw_trajectory(pts, mask, canvas, pixel_overlap, x_max, y_max, bev_reso):
        coords_y = (y_max - pts[:, 1]) / bev_reso[1]
        coords_x = (x_max + pts[:, 0]) / bev_reso[0]
        coords = np.stack([coords_x, coords_y], axis=-1).astype(np.int32)
        if len(coords) <= 1:
            return

        for i in range(len(coords) - 1):
            if np.all(mask[i] == False) or np.all(mask[i + 1] == False):
                continue
            t = i / (len(coords) - 1)
            color = interpolate_color(ColorBoard.light_blue, ColorBoard.blue, t)
            cv2.line(canvas, tuple(coords[i]), tuple(coords[i + 1]), color=color, thickness=3)
            cv2.line(pixel_overlap, tuple(coords[i]), tuple(coords[i + 1]), color=(1, 1, 1), thickness=3)

    @staticmethod
    def draw_prediction_map(objs, masks, top, left, bev_reso, num_future_frames, canvas):

        canvas_float = np.zeros_like(canvas, dtype=np.float32)
        pixel_overlap_float = np.zeros_like(canvas, dtype=np.float32)
        for (obj, mask) in zip(objs, masks):
            trajectory = obj.reshape(-1, 2)  # T,2
            mask_bool = mask.astype(bool)
            canvas = np.zeros(canvas_float.shape, dtype=np.uint8)
            pixel_overlap = np.zeros(pixel_overlap_float.shape, dtype=np.uint8)
            Rasterize.draw_trajectory(trajectory, mask_bool, canvas, pixel_overlap, top, left, bev_reso)
            canvas_float = canvas_float + canvas
            pixel_overlap_float = pixel_overlap_float + pixel_overlap

        eps = 1e-9
        canvas = (canvas_float / (pixel_overlap_float + eps)).astype(np.uint8)
        return canvas

    def draw_planning_map(pts, x_max, y_max, bev_reso, canvas):
        coords_y = (y_max - pts[:, 1]) / bev_reso[1]
        coords_x = (x_max + pts[:, 0]) / bev_reso[0]
        coords = np.stack([coords_x, coords_y], axis=-1).astype(np.int32)

        for i, point in enumerate(coords):
            t = i / (len(coords) - 1)
            color = interpolate_color(ColorBoard.orange, ColorBoard.pink, t)
            cv2.circle(canvas, point, radius=3, color=color, thickness=-1)

        return canvas
