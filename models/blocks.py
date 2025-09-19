import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard convolutional block with GroupNorm, SiLU activation,
    a residual connection, and time embedding injection.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, is_res=True):
        super().__init__()
        self.is_res = is_res
        self.main_path = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.main_path(x)
        time_emb = self.time_mlp(temb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        return h + self.res_conv(x)

def sinusoidal_embedding(timesteps: torch.LongTensor, dim: int):
    """
    timesteps: (B,) long tensor
    returns: (B, dim) float tensor
    """
    assert len(timesteps.shape) == 1
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(half, device=device) / (half - 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class AttentionBlock(nn.Module):
    """
    Self-attention block. Applies Multi-Head Self-Attention to 2D feature maps.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)

        attn = torch.einsum('b h c i, b h c j -> b h i j', q, k) * ((C // self.num_heads) ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('b h i j, b h c j -> b h c i', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)