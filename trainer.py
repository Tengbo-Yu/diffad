import os
from collections import OrderedDict
import functools
from copy import deepcopy
from einops import rearrange
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers.models import AutoencoderKL
from dataset.nuscenes_e2e_dataset import NuScenesE2EDataset
from dataset.b2d_e2e_dataset import B2DE2EDataset
from dataset.util.collate import collate_3d
from model.ddpm_diffad import DDPMDiffad
from model.fm_diffad import FMDiffad
from util.dist_util import (
    cleanup,
    init_dist,
    setup_logger,
    set_seed
)
from util.flatten_util import (
    flatten_sequential_data,
    select_data_dict_by_idx,
    flatten_and_normalize_map,
    to_device
)
from util.visualize_util import BevVisualizer
from util.load_and_save_callback import LoadAndSaveCallback
from util.freeze_callback import FreezeCallback
from util.dump_util import dump_image, dump_dict


class DiffadTrainer:
    def __init__(self, config, data_type):
        self.rank, self.device = init_dist()

        self.config = config
        self.global_config = config['Global']
        self.train_config = config['Train']
        self.dataset_config = config['Dataset']
        self.visualizer = BevVisualizer(config)
        set_seed(seed=self.global_config['global_seed'])
        self.logger, self.writer = setup_logger(self.global_config, self.rank)

        self.step = 0
        self.epoch = 0
        self.total_epoch = self.train_config['max_epoch']
        self.log_interval = self.train_config['log_every_step']
        self.save_interval = self.train_config['save_every_step']

        # load data
        self.data_loader = self._load_data(data_type)

        # bev+obstacle+motion RGB map
        self.vae_config = config["VAE"]
        self.vae_in_channels = self.vae_config['latent_channels'] * 3
        self.vae_latent_size = self.vae_config['latent_size']
        vae_model_path = self.vae_config['model_path']
        self.vae = AutoencoderKL.from_pretrained(vae_model_path).to(self.device)
        self.vae.eval()

        # init diffad model
        if self.global_config['diffusion_type'] == 'dpm':
            diffad = DDPMDiffad(config, num_inference_steps=10)
        elif self.global_config['diffusion_type'] == 'fm':
            diffad = FMDiffad(config, num_inference_steps=4)
        else:
            raise ValueError('invalid diffusion type!')
        diffad.build_diffad()
        diffad = diffad.to(self.device)
        self.ema = deepcopy(diffad)
        # setup optimizer
        self.opt = self._build_optimizer(diffad)
        if self.rank == 0:
            load_from = self.global_config['load_from']
            load_save_func = LoadAndSaveCallback(load_from, ignore_keys=[])
            load_save_func.load_diffad(diffad)
            load_save_func.load_ema(self.ema)
            self.step = find_resume_step(load_from)

        # process freeze parameters
        freeze_func = FreezeCallback()
        freeze_func.freeze_before_training(diffad)
        self.requires_grad(self.ema, False)
        dist.barrier()

        self.ddp_model = DDP(diffad, device_ids=[self.device])  # local_rank
        self.logger.info(f"LDM Parameters: {sum(p.numel() for p in self.ddp_model.parameters()) / 1000 / 1000 :.1f} M")

    def _load_data(self, data_type):
        config = self.dataset_config[data_type]
        obj_type = config.pop('type')
        if obj_type == 'NuScenesE2EDataset':
            dataset = NuScenesE2EDataset(**config)
        elif obj_type == 'B2DE2EDataset':
            dataset = B2DE2EDataset(**config)
        else:
            raise 'invalid Dataset type!'

        # if shuffle=True, sampler shuffles the whole dataset before split to rank
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True if data_type == 'train' else False,
            seed=self.global_config['global_seed']
        )
        loader = DataLoader(
            dataset,
            collate_fn=functools.partial(
                collate_3d,
                ignore_keys=['metadata', 'frames']),
            batch_size=self.train_config['image_per_gpu'] if data_type == 'train' else 1,
            shuffle=False,
            sampler=sampler,
            num_workers=self.train_config['worker_per_gpu'],
            pin_memory=True,
            drop_last=data_type == 'train'
        )
        self.logger.info(f"{data_type} dataset contains {len(dataset):,}")
        return loader

    def _build_optimizer(self, torch_module):
        params_group = []
        lr = self.train_config['optimizer']['lr']
        weight_decay = self.train_config['optimizer']['weight_decay']

        for module_name, module in torch_module.model.named_children():
            cur_params = module.parameters()
            params_group.append({
                'params': list(cur_params),
                'lr': lr,
                'weight_decay': weight_decay
            })

        optimizer = torch.optim.AdamW(params_group)
        return optimizer

    @torch.no_grad()
    def update_ema(self, decay):
        ema_params = OrderedDict(self.ema.named_parameters())
        model_params = OrderedDict(self.ddp_model.module.named_parameters())
        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    @staticmethod
    def requires_grad(model, flag):
        for p in model.parameters():
            p.requires_grad = flag

    def save_ckpt(self):
        if self.rank == 0:
            checkpoint = {
                "model": self.ddp_model.module.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.opt.state_dict(),
            }
            save_path = self.config['Global']['save_path']
            checkpoint_path = f"{save_path}/step_{self.step}.pt"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"saved checkpoint to {checkpoint_path}")

    @torch.no_grad()
    def eval_step(self, batch):
        image = batch['image']  # {'cam_id': [b,t,c,h,w]}
        lidar2img = batch['lidar2img']
        lidar2global = batch['lidar2global']
        bs, ts = lidar2global.shape[:2]  # bs,ts,matrix

        # tensor, bs, ts, c,h,w
        perc_map, motion_map, plan_map = map(flatten_and_normalize_map,
                                             [batch['perc_map'],
                                              batch['motion_map'],
                                              batch['plan_map']])

        bev_image_gt = torch.cat([perc_map, motion_map, plan_map], dim=-1)  # bs*ts,c,h,w*3

        plan_traj_vcs_gt = rearrange(batch['sdc_planning'], "B T L D -> (B T) L D", T=ts)

        # Move inputs to device
        task_label_flatten, ego_status_flatten, command_flatten = map(flatten_sequential_data,
                                                                      [batch['task_label'],
                                                                       batch['ego_status'],
                                                                       batch['command']])
        image, lidar2global, task_label_flatten, ego_status_flatten, command_flatten = \
            map(lambda x: to_device(x, self.device),
                [image, lidar2global, task_label_flatten,
                 ego_status_flatten, command_flatten])

        z = torch.randn((bs * ts, self.vae_in_channels, *self.vae_latent_size[1:]), device=self.device)
        samples, extra_samples = self.ddp_model.module.recurrent_sample(z, image, lidar2img, lidar2global,
                                                                        task_label_flatten,
                                                                        ego_status_flatten, command_flatten)

        def decode_samples(samples, num_chunks, bs, ts, vae, scale=0.18215):
            # Divide samples into chunks
            chunks = torch.chunk(samples, chunks=num_chunks, dim=1)
            samples = torch.cat(chunks, dim=0)  # bs*T*N

            # Decode samples
            samples_list = []
            for t_idx in range(ts):
                cur_samples = vae.decode(samples[t_idx * bs * num_chunks:(t_idx + 1) * bs * num_chunks] / scale).sample
                # cur_samples = rearrange(cur_samples, "(B N) C H W -> N B C H W", B=bs, N=num_chunks)
                samples_list.append(cur_samples)
            return torch.cat(samples_list)

        # Channel to bs
        num_chunks = 3
        samples = decode_samples(samples, num_chunks, bs, ts, self.vae)
        fake_bev, fake_mot, fake_plan = torch.chunk(samples, chunks=num_chunks, dim=0)  # TODO: mv chunk into visual
        bev_image_fake = torch.cat([fake_bev, fake_mot, fake_plan], dim=-1)  # bs*ts,c,h,w*3

        plan_traj_vcs_fake = extra_samples.squeeze(1)  # bs*ts,L,D

        return bev_image_gt, plan_traj_vcs_gt, bev_image_fake, plan_traj_vcs_fake

    def log_step(self, loss_dict):
        if dist.get_rank() == 0:
            # Log values
            mse_loss = loss_dict.get("mse").mean().item()
            parse_loss = loss_dict.get("parse_loss").item()
            loss_value = loss_dict["loss"].item()
            loss_and_lr_info = {"mse loss": mse_loss, "parse_loss": parse_loss, "loss": loss_value}
            log_text = f"(epoch={self.epoch}/{self.total_epoch} step={self.step}) "
            log_text += " ".join([f"{key}: {value:.6f}" for key, value in loss_and_lr_info.items()])
            self.logger.info(log_text)

            if self.step % self.log_interval == 0:
                for key, value in loss_and_lr_info.items():
                    self.writer.add_scalar(key, value, self.step)

    def train_loop(self):
        self.update_ema(decay=0)  # Ensure EMA is initialized with synced weights
        self.ddp_model.train()  # important! This enables embedding dropout for classifier-free guidance
        self.ema.eval()  # EMA model should always be in eval mode
        self.logger.info(f"Training for {self.total_epoch} epochs...")
        for _ in range(self.total_epoch):
            self.epoch += 1
            self.logger.info(f"Beginning epoch {self.epoch}...")
            self.data_loader.sampler.set_epoch(self.epoch)
            for batch in self.data_loader:
                self.train_step(batch)
                self.step += 1
                # log loss scalar and visualization to tensorboard
                if self.step % self.log_interval == 0:
                    self.log_vis_step(batch)
                if self.step % self.save_interval == 0:
                    self.save_ckpt()
                    dist.barrier()

        self.save_ckpt()
        self.logger.info("Done!")
        cleanup()

    def train_step(self, batch):
        lidar2img = batch['lidar2img']
        # tensor, bs, ts, c,h,w
        perc_map, motion_map, plan_map = map(flatten_and_normalize_map,
                                             [batch['perc_map'].to(self.device),
                                              batch['motion_map'].to(self.device),
                                              batch['plan_map'].to(self.device)])

        # VAE encoding (without gradient tracking)
        with torch.no_grad():
            vae_inputs = torch.cat([perc_map, motion_map, plan_map], dim=0)
            vae_outputs = self.vae.encode(vae_inputs).latent_dist.sample().mul_(0.18215)
            perc_latent, mot_latent, plan_latent = torch.chunk(vae_outputs, chunks=3, dim=0)  # bs*ts*2,c,h,w
            latent_z = torch.cat([perc_latent, mot_latent, plan_latent], dim=1)  # bs*ts,c*3,h,w,

        # Move inputs to device
        task_label_flatten, ego_status_flatten, command_flatten = map(flatten_sequential_data,
                                                                      [batch['task_label'],
                                                                       batch['ego_status'], batch['command']])
        image, task_label_flatten, ego_status_flatten, command_flatten = map(lambda x: to_device(x, self.device),
                                                                             [batch['image'], task_label_flatten,
                                                                              ego_status_flatten, command_flatten])

        # Prepare trajectory data
        lidar2global = batch['lidar2global']  # bs,ts,matrix
        bs, ts = lidar2global.shape[:2]
        plan_traj_vcs_gt = rearrange(batch['sdc_planning'], "B T (C L) D -> (B T) C L D", C=1, T=ts)
        plan_traj_vcs_mask = rearrange(batch['sdc_planning_mask'], "B T (C L) D -> (B T) C L D", C=1, T=ts)

        model_kwargs = {
            'task_label': task_label_flatten,
            'ego_status': ego_status_flatten,
            'command': command_flatten
        }

        # Compute loss
        loss_dict = self.ddp_model(latent_z, plan_traj_vcs_gt, plan_traj_vcs_mask,
                                   image, lidar2img, lidar2global, model_kwargs)
        loss = loss_dict["loss"].mean()
        loss_dict["loss"] = loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.update_ema(decay=0.9999)
        self.log_step(loss_dict)

    def log_vis_step(self, batch):
        if dist.get_rank() == 0:
            self.ddp_model.eval()
            (bev_image_gt, plan_traj_vcs_gt,
             bev_image_fake, plan_traj_vcs_fake) = self.eval_step(batch)

            metadata = batch['metadata']
            command = flatten_sequential_data(batch['command'])
            lidar2img = flatten_sequential_data(batch['lidar2img'])
            bs, ts = batch['lidar2global'].shape[:2]  # bs,ts,matrix
            bs_ts = bs * ts
            selected_idx = np.arange(bs_ts)
            for show_idx, idx in enumerate(selected_idx):
                cur_lidar2img = select_data_dict_by_idx(lidar2img, idx)
                cur_pred_image = bev_image_fake[idx, ...].detach().cpu().numpy()  # bs*ts,C,H,W
                cur_gt_image = bev_image_gt[idx, ...].detach().cpu().numpy()
                cur_gt_plan_traj_vcs = plan_traj_vcs_gt[idx, ...].cpu().numpy()
                cur_pred_plan_traj_vcs = plan_traj_vcs_fake[idx, ...].cpu().numpy()
                cur_command = command[idx, ...].cpu().numpy()
                cur_metadata = metadata[idx // ts]['frame_metadata'][idx % ts]
                vis_img = self.visualizer.visualize(cur_metadata['filename'], cur_lidar2img, cur_pred_image,
                                                    cur_gt_image, cur_gt_plan_traj_vcs, cur_pred_plan_traj_vcs,
                                                    cur_command)
                vis_name = "training_step={0}/data_name={1}/vis_idx={2}/clip_id={3}/ts={4}".format(
                    self.step, 'e2e', show_idx, cur_metadata["clip_id"], cur_metadata["timestamp"])
                self.writer.add_image(vis_name, vis_img, self.step)

            self.ddp_model.train()

    def eval_loop(self):
        self.ddp_model.eval()
        self.ema.eval()
        root_dump_path = os.path.join(self.global_config['save_path'], 'eval')
        os.makedirs(root_dump_path, exist_ok=True)
        for batch in tqdm(self.data_loader):
            (bev_image_gt, plan_traj_vcs_gt,
             bev_image_fake, plan_traj_vcs_fake) = self.eval_step(batch)

            bs, ts = batch['lidar2global'].shape[:2]  # bs,ts,matrix
            lidar2img = flatten_sequential_data(batch['lidar2img'])
            command = flatten_sequential_data(batch['command'])
            metadata = batch['metadata']
            for b_idx in range(bs):
                for t_idx in range(ts):
                    idx = b_idx * ts + t_idx
                    cur_lidar2img = select_data_dict_by_idx(lidar2img, idx)
                    cur_bev_image_fake = bev_image_fake[idx, ...].detach().cpu().numpy()  # bs*ts,C,H,W
                    cur_bev_image_gt = bev_image_gt[idx, ...].detach().cpu().numpy()
                    cur_plan_traj_vcs_gt = plan_traj_vcs_gt[idx, ...].cpu().numpy()
                    cur_plan_traj_vcs_fake = plan_traj_vcs_fake[idx, ...].cpu().numpy()
                    cur_command = command[idx, ...].cpu().numpy()
                    cur_metadata = metadata[b_idx]['frame_metadata'][t_idx]

                    # dump pred json
                    eval_dump_path = os.path.join(root_dump_path, 'eval_log', cur_metadata['clip_id'])
                    os.makedirs(eval_dump_path, exist_ok=True)
                    eval_data = {
                        'expert_traj': cur_plan_traj_vcs_gt,
                        'planner_traj': cur_plan_traj_vcs_fake,
                    }
                    eval_dump_name = "{0}_{1}.json".format(cur_metadata['timestamp'], cur_metadata['frame_id'])
                    dump_dict(os.path.join(eval_dump_path, eval_dump_name), eval_data)

                    # dump image
                    cur_vis_image = self.visualizer.visualize(cur_metadata['filename'], cur_lidar2img,
                                                              cur_bev_image_fake, cur_bev_image_gt,
                                                              cur_plan_traj_vcs_gt, cur_plan_traj_vcs_fake, cur_command)
                    image_dump_name = "{0}_{1}.png".format(cur_metadata['timestamp'], cur_metadata['frame_id'])

                    image_dump_path = os.path.join(root_dump_path, 'image', cur_metadata['clip_id'])
                    os.makedirs(image_dump_path, exist_ok=True)
                    dump_image(os.path.join(image_dump_path, image_dump_name), cur_vis_image)

        self.logger.info("Done!")
        cleanup()


def find_resume_step(load_from):
    # path/to/model/NNN.pt
    split_number = load_from.split('/')[-1].split(".")[0]
    try:
        return int(split_number)
    except ValueError:
        return 0
