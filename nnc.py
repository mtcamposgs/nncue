# =============================================
#  NNCUE: Efficiently Updatable Complex Neural 
#         Networks for Computer Chess
#      Using NNC (Complex Neural Network)
# ---------------------------------------------
#  By: Matheus Campos               12/02/2025
# =============================================

# This code is a simplification. Understand how
# the architecture works if you need a more
# complete one.

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_norm

class ResSwiGTU_Block(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.projector = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        nn.init.xavier_uniform_(self.projector.weight, gain=1.0)
    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        mixed = self.projector(x_norm)
        left, right = mixed.chunk(2, dim=-1)
        tension = left - right
        context = left + right
        gtu_out = tension * F.silu(context)
        return residual + gtu_out

class NNCUE_Network(nn.Module):
    def __init__(self, num_inputs=768, hidden_dim=256, num_layers=2):
        super().__init__()
        self.feature_extractor = nn.EmbeddingBag(num_inputs, 512, mode='sum')
        self.compressor = nn.Linear(512, hidden_dim)
        self.layers = nn.ModuleList([
            ResSwiGTU_Block(hidden_dim) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.output_head.weight, gain=0.1)
        nn.init.zeros_(self.output_head.bias)
    def forward(self, input_indices, input_offsets):
        x = self.feature_extractor(input_indices, input_offsets)
        x = F.silu(self.compressor(x))
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        score = self.output_head(x)
        return score