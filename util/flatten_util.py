import torch


def unflatten_sequential_data(data, bs, ts):
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = unflatten_sequential_data(val, bs, ts)
    elif isinstance(data, torch.Tensor):
        _, *feature_dimensions = data.shape
        data = data.reshape(bs, ts, *feature_dimensions)
    else:
        raise TypeError(f"[unflatten_sequential_data] unsupported type {type(data)}")
    return data


def flatten_sequential_data(data):
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = flatten_sequential_data(val)
    elif isinstance(data, torch.Tensor):
        batch, time, *feature_dimensions = data.shape
        data = data.reshape(batch * time, *feature_dimensions)
    else:
        raise TypeError(f"[flatten_sequential_data] unsupported type {type(data)}")
    return data


def to_device(data, device):
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = to_device(val, device)
    elif isinstance(data, list):
        new_list = []
        for val in data:
            new_list.append(to_device(val, device))
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    else:
        raise TypeError(f"[unflatten_sequential_data] unsupported type {type(data)}")
    return data


def flatten_and_normalize_map(maps):
    assert isinstance(maps, torch.Tensor)
    maps = flatten_sequential_data(maps)
    maps = ((maps - 128) / 128).clamp(-1, 1)
    return maps


def select_data_dict_by_idx(data_dict, idx, to_cpu=True):
    result = {}
    for name, data in data_dict.items():
        result[name] = data[idx, ...].detach().cpu().numpy() if to_cpu else data[idx, ...]
    return result


def select_data_dict_by_t_idx(data_dict, t_idx, to_cpu=True):
    result = {}
    for name, data in data_dict.items():
        result[name] = data[:, t_idx, ...].detach().cpu().numpy() if to_cpu else data[:, t_idx, ...]
    return result
