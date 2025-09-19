import torch
from torch import nn
import torch.nn.functional as F

class Diffusion:
    def __init__(self, model: nn.Module, img_size=96, device="cuda", timesteps=256):
        self.model = model
        self.img_size = img_size
        self.device = device
        self.timesteps = timesteps

        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    def q_sample(self, x0, t, noise=None):
        """
        sample from q(x_t | x_0)
        x0: (B,C,H,W)
        t: (B,) long tensor
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    def training_step(self, x0, cond):
        """
        single batch training step: pick random t for each sample
        returns MSE loss between predicted noise and real noise
        """
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t, cond)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def p_mean_variance(self, x_t, t, cond):
        """
        computes posterior mean and variance for q(x_{t-1} | x_t, x0_pred)
        x_t: (B,C,H,W)
        t: scalar int or 0-d python int
        returns: posterior_mean, posterior_variance (both tensors shape (B,C,H,W) for mean and (B,1,1,1) for var)
        """
        B = x_t.shape[0]
        device = x_t.device
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps_theta = self.model(x_t, t_tensor, cond)

        sqrt_alpha_hat_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x_t - sqrt_one_minus_alpha_hat_t * eps_theta) / sqrt_alpha_hat_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        alpha_t = self.alphas[t]
        alpha_hat_t = self.alphas_cumprod[t]
        alpha_hat_prev = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]

        coef_x0 = (torch.sqrt(alpha_hat_prev) * beta_t) / (1.0 - alpha_hat_t)
        coef_xt = (torch.sqrt(alpha_t) * (1.0 - alpha_hat_prev)) / (1.0 - alpha_hat_t)

        coef_x0 = coef_x0.view(1, 1, 1, 1)
        coef_xt = coef_xt.view(1, 1, 1, 1)

        posterior_mean = coef_x0 * x0_pred + coef_xt * x_t
        posterior_variance = beta_t * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20)).view(1, 1, 1, 1)

        return posterior_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x_t, t, cond):
        """
        sample x_{t-1} from p(x_{t-1} | x_t)
        """
        mean, var, log_var = self.p_mean_variance(x_t, t, cond)
        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var).view(1,1,1,1) * noise

    def sample(self, cond, batch_size=8):
        """
        Full sampling loop: start from x_T ~ N(0,I), run p_sample iteratively.
        """
        x = torch.randn(batch_size, 3, self.img_size, self.img_size, device=self.device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, cond)
        return x