import torch
import numpy as np
from tqdm import tqdm
from ldm.modules.diffusionmodules.util import make_beta_schedule

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith('ddim'):
            desired_count = int(section_counts[len('ddim'):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f'cannot create exactly {num_timesteps} steps with an integer stride')
        section_counts = [int(x) for x in section_counts.split(',')]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for (i, section_count) in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f'cannot divide section of {size} steps into {section_count}')
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class SpacedSampler:

    def __init__(self, model, schedule='linear', var_type: str='fixed_small'):
        self.model = model
        self.original_num_steps = model.num_timesteps
        self.used_num_steps = model.used_timesteps
        self.schedule = schedule
        self.var_type = var_type

    def make_schedule(self, num_steps):
        original_betas = make_beta_schedule(self.schedule, self.original_num_steps, linear_start=self.model.linear_start, linear_end=self.model.linear_end)
        original_alphas = 1.0 - original_betas
        original_alphas_cumprod = np.cumprod(original_alphas, axis=0)
        used_timesteps = space_timesteps(self.used_num_steps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for (i, alpha_cumprod) in enumerate(original_alphas_cumprod):
            if i in used_timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        assert len(betas) == num_steps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (num_steps,)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        assert posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0]
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    @torch.no_grad()
    def sample(self, steps, shape, conditioning=None, x_T=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None, cond_fn=None):
        self.make_schedule(num_steps=steps)
        samples = self.sapced_sampling(conditioning, shape, x_T=x_T, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, cond_fn=cond_fn)
        return samples

    def sample_grad(self, steps, shape, conditioning=None, x_T=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None, cond_fn=None):
        self.make_schedule(num_steps=steps)
        samples = self.sapced_sampling_grad(conditioning, shape, x_T=x_T, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, cond_fn=cond_fn)
        return samples

    @torch.no_grad()
    def sapced_sampling(self, cond, shape, x_T, unconditional_guidance_scale, unconditional_conditioning, cond_fn):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        time_range = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        for (i, step) in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            img = self.p_sample_spaced(img, cond, ts, index=index, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, cond_fn=cond_fn)
        return img

    def sapced_sampling_grad(self, cond, shape, x_T, unconditional_guidance_scale, unconditional_conditioning, cond_fn):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        time_range = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        for (i, step) in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            img = self.p_sample_spaced_grad(img, cond, ts, index=index, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, cond_fn=cond_fn)
        return img

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def predict_noise(self, x, t, c, unconditional_guidance_scale, unconditional_conditioning):
        if unconditional_conditioning is None and unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c)
        else:
            model_t = self.model.apply_model(x, t, c)
            model_uncond = self.model.apply_model_unconditional(x, t, c)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
        if self.model.parameterization == 'v':
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output
        return e_t

    def apply_cond_fn(self, x, c, t, index, cond_fn, unconditional_guidance_scale, unconditional_conditioning):
        device = x.device
        t_now = int(t[0].item()) + 1
        e_t = self.predict_noise(x, t, c, unconditional_guidance_scale, unconditional_conditioning)
        pred_x0: torch.Tensor = self._predict_xstart_from_eps(x_t=x, t=index, eps=e_t)
        (model_mean, _, _) = self.q_posterior_mean_variance(x_start=pred_x0, x_t=x, t=index)
        for _ in range(cond_fn.repeat):
            (target, pred) = (None, None)
            if cond_fn.space == 'latent':
                target = self.model.get_first_stage_encoding(self.model.encode_first_stage(cond_fn.target.to(device)))
                pred = pred_x0
            elif cond_fn.space == 'rgb':
                with torch.enable_grad():
                    pred_x0.requires_grad_(True)
                    target = cond_fn.target.to(device)
                    pred = self.model.decode_first_stage_with_grad(pred_x0)
            else:
                raise NotImplementedError(cond_fn.space)
            delta_pred = cond_fn(target, pred, t_now)
            if delta_pred is not None:
                if cond_fn.space == 'rgb':
                    pred.backward(delta_pred)
                    delta_pred_x0 = pred_x0.grad
                    pred_x0 += delta_pred_x0
                    model_mean += 0.5 * delta_pred_x0
                    pred_x0.grad.zero_()
                else:
                    delta_pred_x0 = delta_pred
                    pred_x0 += delta_pred_x0
                    model_mean += 0.5 * delta_pred_x0
            else:
                break
        return (model_mean.detach().clone(), pred_x0.detach().clone())

    @torch.no_grad()
    def p_sample_spaced(self, x: torch.Tensor, c, t, index, unconditional_guidance_scale, unconditional_conditioning, cond_fn):
        index = torch.full_like(t, fill_value=index)
        model_variance = {'fixed_large': np.append(self.posterior_variance[1], self.betas[1:]), 'fixed_small': self.posterior_variance}[self.var_type]
        model_variance = _extract_into_tensor(model_variance, index, x.shape)
        if cond_fn is not None:
            (model_mean, pred_x0) = self.apply_cond_fn(x, c, t, index, cond_fn, unconditional_guidance_scale, unconditional_conditioning)
        else:
            e_t = self.predict_noise(x, t, c, unconditional_guidance_scale, unconditional_conditioning)
            pred_x0 = self._predict_xstart_from_eps(x_t=x, t=index, eps=e_t)
            (model_mean, _, _) = self.q_posterior_mean_variance(x_start=pred_x0, x_t=x, t=index)
        noise = torch.randn_like(x)
        nonzero_mask = (index != 0).float().view(-1, *[1] * (len(x.shape) - 1))
        x_prev = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return x_prev

    def p_sample_spaced_grad(self, x: torch.Tensor, c, t, index, unconditional_guidance_scale, unconditional_conditioning, cond_fn):
        index = torch.full_like(t, fill_value=index)
        model_variance = {'fixed_large': np.append(self.posterior_variance[1], self.betas[1:]), 'fixed_small': self.posterior_variance}[self.var_type]
        model_variance = _extract_into_tensor(model_variance, index, x.shape)
        if cond_fn is not None:
            (model_mean, pred_x0) = self.apply_cond_fn(x, c, t, index, cond_fn, unconditional_guidance_scale, unconditional_conditioning)
        else:
            e_t = self.predict_noise(x, t, c, unconditional_guidance_scale, unconditional_conditioning)
            pred_x0 = self._predict_xstart_from_eps(x_t=x, t=index, eps=e_t)
            (model_mean, _, _) = self.q_posterior_mean_variance(x_start=pred_x0, x_t=x, t=index)
        noise = torch.randn_like(x)
        nonzero_mask = (index != 0).float().view(-1, *[1] * (len(x.shape) - 1))
        x_prev = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return x_prev
