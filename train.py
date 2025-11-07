import argparse
from util.config_util import update_config, load_config
from trainer import DiffadTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file path')
    parser.add_argument('--diffusion_type', type=str, default='dpm', required=False,
                        help='diffusion type [dpm, fm]')
    parser.add_argument('--ckpt', type=str, default='', required=False,
                        help='checkpoint')

    args = parser.parse_args()
    config = load_config(args, phase='training')
    config['diffusion_type'] = args.diffusion_type
    config['load_from'] = args.ckpt
    update_config(config)
    trainer = DiffadTrainer(config, data_type='train')
    trainer.train_loop()
