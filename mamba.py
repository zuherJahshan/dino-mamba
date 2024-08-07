import os 

import torch
from fairseq.models.wav2vec import TransformerEncoder
from mamba_ssm import Mamba

from dinosr_config import DinosrAudioConfig
from einops import rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch.nn as nn

class DeepMamba(nn.Module):
    def __init__(
        self,
        cfg: DinosrAudioConfig,
    ):
        self.num_layers = cfg.encoder_layers
        self.d_model = cfg.encoder_embed_dim
        self.d_state = cfg.mamba_d_state

        super(DeepMamba, self).__init__()
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=4,
                expand=2
        ) for _ in range(self.num_layers)])
        self.lns = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.num_layers)
        ])

    def forward(self, x):
        layer_results = []
        for mamba, ln in zip(self.mamba_layers, self.lns):
            x = mamba(ln(x))
            layer_results.append(([], [], rearrange(x, 'b t d -> t b d')))
        return rearrange(x, 'b t d -> t b d'), layer_results

# %%
# dinosr_cfg = DinosrAudioConfig()

# %%
# model = DeepMamba(dinosr_cfg).to(device)

# B, T, d = 1, 10, dinosr_cfg.encoder_embed_dim
# x = torch.randn(B, T, d).to(device)
# y, layer_results = model(x)

# %%
# y.shape

# %%



