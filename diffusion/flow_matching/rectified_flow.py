import torch
from diffusion.flow_matching.path.scheduler.scheduler import CondOTScheduler
from diffusion.flow_matching.path.affine import AffineProbPath
from diffusion.flow_matching.solver.ode_solver import ODESolver
from diffusion.flow_matching.utils.model_wrapper import ModelWrapper


class ReflowDiffusion:
    def __init__(self):
        self.path = AffineProbPath(scheduler=CondOTScheduler())

    def training_losses(self, model, x_1, t, model_kwargs=None):
        x_0 = torch.randn_like(x_1).to(x_1.device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # flow matching l2 loss
        model_output = model(path_sample.x_t, path_sample.t, **model_kwargs)
        loss = torch.pow(model_output - path_sample.dx_t, 2).mean()
        terms = {'loss': loss, 'mse': loss}
        return terms

    def solver_sample(self, model, x_0, model_kwargs=None):
        T = torch.linspace(0, 1, self.num_inference_steps)  # sample times
        T = T.to(x_0.device)
        wrapped_vf = _WrappedModel(model)
        solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
        sample = solver.sample(time_grid=T,
                               x_init=x_0,
                               method='euler',
                               step_size=None,
                               return_intermediates=False,
                               model_extras=model_kwargs)  # sample from the model
        return sample

    def set_infer_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps


class _WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        t_batch = torch.tensor([t] * len(x), device=x.device)
        return self.model(x, t_batch, **extras['model_extras'])
