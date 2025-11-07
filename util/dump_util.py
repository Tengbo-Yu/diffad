import numpy as np
import cv2
import json


def dump_image(save_path, image):
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)


def dump_dict(save_path, obj):
    with open(save_path, "w") as f:
        json.dump(obj, f, cls=JsonSerializer)


def dump_str(save_path, content):
    with open(save_path, "w") as f:
        f.writelines(content.encode('ascii', 'ignore').decode('ascii'))


class JsonSerializer(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            o = o.squeeze()
            shape = o.shape
            o = np.array([float(obj) for obj in o.flatten()]).reshape(shape)
            o = o.tolist()
        elif isinstance(o, list):
            o = [float(obj) for obj in o]
        elif isinstance(o, (str, bool)):
            o = json.JSONEncoder.default(self, o)
        else:
            o = float(o)
        return o
