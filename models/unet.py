import torch
from torch import nn

from models.blocks import ConvBlock, AttentionBlock, sinusoidal_embedding


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=128, time_emb_dim=256, cond_dim = 47):
        super().__init__()

        # Sinusoidalne embeddingi
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Wektor cech
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )


        ch_mults = (1, 2, 4, 8)
        channels = [base_channels] + [base_channels * m for m in ch_mults]

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        for i in range(len(ch_mults)):
            in_ch = channels[i]
            out_ch = channels[i+1]
            self.down_blocks.append(nn.ModuleList([
                ConvBlock(in_ch, out_ch, time_emb_dim),
                AttentionBlock(out_ch) if i >= 2 else nn.Identity(),
                nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
            ]))

        self.mid_block1 = ConvBlock(channels[-1], channels[-1], time_emb_dim)
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = ConvBlock(channels[-1], channels[-1], time_emb_dim)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(ch_mults))):
            in_ch = channels[i+1]
            out_ch = channels[i]

            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                ConvBlock(out_ch + in_ch, out_ch, time_emb_dim),
                AttentionBlock(out_ch) if i >= 2 else nn.Identity(),
            ]))

        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t, cond):
        temb = sinusoidal_embedding(t, 256)
        temb = self.time_mlp(temb)

        cemb = self.cond_mlp(cond)
        temb = temb + cemb
        x = self.init_conv(x)

        skips = [x]
        for block, attn, downsample in self.down_blocks:
            x = block(x, temb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, temb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, temb)

        for upsample, block, attn in self.up_blocks:
            x = upsample(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, temb)
            x = attn(x)

        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)