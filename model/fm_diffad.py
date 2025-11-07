import torch
from model.base_diffad import BaseDiffad
from diffusion import create_fm_diffusion


class FMDiffad(BaseDiffad):
    def __init__(self, config, num_inference_steps):
        super().__init__(config)
        self.diffusion = create_fm_diffusion(fm_type='reflow')
        self.diffusion.set_infer_timesteps(num_inference_steps)

    def forward(self, latent_z, plan_traj, plan_traj_mask, image, lidar2img, lidar2global, model_kwargs):
        """
            latent_z(bs*ts,c,h,w): x_start
        """
        prev_x_start = latent_z.clone()
        bevfeature, prop_prev_x_start = self.bevnet_forward(image, lidar2img, lidar2global, prev_x_start)
        model_kwargs.update({'bevfeature': bevfeature, 'prop_prev_x_start': prop_prev_x_start})
        # sample t
        bs, ts = lidar2global.shape[:2]
        t = torch.rand(bs).to(latent_z.device)
        t = t.unsqueeze(1).repeat(1, ts).flatten(0, 1)  # bs*ts
        loss_dict = self.diffusion.training_losses(self.model['ldm'], latent_z, t, model_kwargs)

        if 'postprocess' in self.model:
            parse_loss = self.model['postprocess'].loss_func(latent_z, plan_traj, plan_traj_mask)
            loss_dict['parse_loss'] = parse_loss
            loss_dict['loss'] = loss_dict['loss'] + parse_loss
        else:
            loss_dict['parse_loss'] = torch.tensor([0.]).to(latent_z.device)
        return loss_dict

    def sample_single_step(self, cur_image, cur_lidar2img, cur_lidar2global, prev_h, prev_c, prev_x_start,
                           cur_z, model_kwargs):
        prev_h, prev_c, prop_prev_x_start = self.bevnet_recurrent_forward(
            cur_image, cur_lidar2img, cur_lidar2global, prev_h, prev_c, prev_x_start
        )
        model_kwargs.update({
            'bevfeature': prev_h,
            'prop_prev_x_start': prop_prev_x_start
        })
        samples = self.diffusion.solver_sample(self.model['ldm'], cur_z, model_kwargs=model_kwargs)
        # extract trajectory from bev latent
        pred_trajs = self.model['postprocess'](samples) if 'postprocess' in self.model \
            else torch.zeros(cur_z.shape[0], 1, 12, 2)
        return samples, pred_trajs, prev_h, prev_c
