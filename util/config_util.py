import os
import yaml


def is_rank_zero():
    node_rank = int(os.getenv("RANK", 0))
    if node_rank == 0:
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        return local_rank == 0
    return False


def update_config(config):
    # update global config
    global_config = config['Global']
    global_config['log_dir'] = f"{global_config['save_path']}/log"
    global_config['vis_dir'] = f"{global_config['save_path']}/vis"
    # update vae config
    if 'VAE' in config:
        vae_config = config['VAE']
        bev_reso = global_config['bev_reso']
        point_cloud_range = global_config['point_cloud_range']
        patch_h = point_cloud_range[4] - point_cloud_range[1]
        patch_w = point_cloud_range[3] - point_cloud_range[0]
        input_img_size = (int(patch_h / bev_reso[1]), int(patch_w / bev_reso[0]))
        vae_config['input_img_size'] = input_img_size
        downsample_ratio = vae_config['downsample_ratio']
        assert input_img_size[0] % downsample_ratio == 0 and input_img_size[1] % downsample_ratio == 0
        latent_dim = config['VAE']['latent_channels']
        latent_size = [latent_dim, input_img_size[0] // downsample_ratio, input_img_size[1] // downsample_ratio]
        vae_config['latent_size'] = latent_size


def load_yaml(yaml_file):
    with open(yaml_file, "r") as infile:
        config = yaml.safe_load(infile)
    return config


def load_config(args, phase):
    config_path = args.config
    config = load_yaml(config_path)

    if phase == 'training' and is_rank_zero:
        save_config(config)
        check_config(config)
    return config


def check_config(config):
    if 'Temporal' in config:
        queue_length = config['Global']['queue_length']
        assert queue_length > 1, f'temporal setting required length > 1 but got {queue_length}'


def save_config(config):
    save_path = config['Global']['save_path']
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), "w") as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False)
