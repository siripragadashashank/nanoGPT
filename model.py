import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # q, k, v all in one causal attention
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout

        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        b, t, c = x.shape

        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head)  # b, t, nh, hs
        q = q.transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head)
        k = k.transpose(1, 2)




