import functools
import pdb
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel

from .config import GIVTConfig

""" debug """


def debug_func(func):
    """Decorator to debug a function when error is encountered."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print(
                "Error encountered! Starting debug session from the beginning of the function."
            )
            pdb.runcall(func, *args, **kwargs)

    return wrapper


class RMSNorm(nn.Module):
    pass


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        assert (
            hidden_dim % num_heads
        ), f"hidden_dim: {hidden_dim} must be divisible by num_heads: {num_heads}"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        residual = x
        x = self.norm.forward(x)

        q, k, v = self.qkv_proj.forward(x).chunk(3)  # (b, s, d)
        q, k, v = map(
            lambda t: rearrange(
                t, "b s (nh hs) -> b s nh hd ", nh=self.num_heads, hs=self.head_dim
            )
        )

        # rotary pos
        q, k = "rotary_emb"
        q, k, v = map(lambda t: rearrange(t, "b s nh hd -> b nh s hd"))

        x = F.scaled_dot_product_attention(q    , k, v, is_causal=True)
        x = rearrange(x, "b nh s hd -> b s (nh hd)")

        x = self.o_proj.forward(x)
        return x + residual


class Mlp(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
    ):
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.norm.forward(x)

        out = self.

class Block(nn.Module):
    def __init__(self, config: GIVTConfig):
        super().__init__(config)

        pass


class GIVTModel(nn.Module):
    def __init__(
        self,
        config: GIVTConfig,
    ):
        super().__init__(config)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        for b in self.blocks:
            b: Block
            x = b.forward(x)
        return x


""" main """


@dataclass
class GIVTTrainingOutput:
    loss: Tensor
    info: dict | None = None


class GIVT(PreTrainedModel):
    config_class = GIVTConfig

    def __init__(self, config: GIVTConfig):
        super().__init__(config)

        self.embed = nn.Linear(config.input_dim, config.hidden_dim, bias=False)
        self.model = GIVTModel(config)
        self.head = nn.Linear(config.hidden_dim, config.input_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        return_info: bool = True,
    ) -> Tensor:
        inputs = x[1:]
        targets = x[:-1].contiguous().float()

        hidden = self.model.forward(inputs)
        predicted = self.head.forward(hidden).contiguous().float()

        loss = F.nll_loss(predicted, targets).float().mean()

        info = None
        if return_info:
            info = dict(variance=predicted.chunk(2)[1].mean())

        return GIVTTrainingOutput(loss=loss, info=info)

    def generate(
        self,
        prompt: Tensor,
        temperature: float = 0.95,
        cfg_scale: float = 0.4,
    ) -> Tensor:
        pass
