import torch
import numpy as np
from PIL import Image
from typing import Any, List, Sequence


def collate_3d(batch_data: List[Any], ignore_keys: Sequence[str] = ()):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    * If output tensor from dataset shape is (n,c,h,w),concat on \
        aixs 0 directly. \
    * If output tensor from dataset shape is (c,h,w),expand_dim on \
        axis 0 and concat.

    Args:
        batch (list): list of data.
        ignore_keys(Sequence): ignore keys in collate_3d
    """
    if isinstance(batch_data[0], dict):
        result = {}
        for key in batch_data[0].keys():
            if key in ignore_keys:
                result[key] = [d[key] for d in batch_data]
            else:
                result[key] = collate_3d(
                    [d[key] for d in batch_data], ignore_keys
                )
        return result
    elif isinstance(batch_data[0], (list, tuple)):
        return [collate_3d(data, ignore_keys) for data in zip(*batch_data)]
    elif isinstance(batch_data[0], torch.Tensor):
        return torch.stack(batch_data, dim=0).type(torch.float32)
    elif isinstance(batch_data[0], (str, int, float, Image.Image)):
        return batch_data
    elif isinstance(batch_data[0], np.ndarray):
        if batch_data[0].dtype.kind == 'U':
            return np.stack(batch_data)
        else:
            return torch.stack([torch.from_numpy(data).type(torch.float32) for data in batch_data], dim=0)
    else:
        raise TypeError
