import functools
import pdb
from pdb import set_trace as bp
from dataclasses import dataclass
from tqdm import tqdm

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


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: torch.device | None = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> tuple[Tensor, Tensor]:
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    head_dim = x.size(-1)
    x1 = x[..., : head_dim // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_dim // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def build_mask_cache(max_seq_length: int, device: torch.device | None = None) -> Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.register_buffer(
            "k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False
        )

    def __repr__(self):
        return f"KVCache(k={self.k.size()}, v={self.v.size()})"

    def forward(self, input_pos: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v

    def reset_parameters(self):
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.weight * x_normed).to(dtype=dtype)


class Attention(nn.Module):
    def __init__(
        self,
        config: GIVTConfig,
    ):
        super().__init__()
        self.config = config
        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.norm = RMSNorm(config.hidden_dim)

        self.kv_cache: KVCache | None = None

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor | None = None,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        residual = x
        x = self.norm.forward(x)

        q, k, v = self.qkv_proj.forward(x).chunk(3, dim=-1)  # (b, s, d)
        q, k, v = map(
            lambda t: rearrange(
                t,
                "b s (nh hd) -> b nh s hd ",
                nh=self.config.num_heads,
                hd=self.config.head_dim,
            ),
            (q, k, v),
        )

        # rotary pos
        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        # kv cache
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `model.set_kv_cache()`")
            k, v = self.kv_cache.forward(input_pos, k, v)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None
        )
        x = rearrange(x, "b nh s hd -> b s (nh hd)")

        x = self.o_proj.forward(x)
        return x + residual

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        v_shape = (
            batch_size,
            self.config.num_heads,
            max_seq_length,
            self.config.head_dim,
        )
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                self.config.num_heads,
                max_seq_length,
                rope_cache_length + self.config.head_dim - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class Mlp(nn.Module):
    def __init__(
        self,
        config: GIVTConfig,
    ):
        super().__init__()
        hidden_dim = config.hidden_dim
        intermediate_dim = config.intermediate_dim
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.fc3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm.forward(x)
        x = F.silu(self.fc1.forward(x)) * self.fc2.forward(x)
        x = self.fc3.forward(x)
        return x + residual


class Block(nn.Module):
    def __init__(self, config: GIVTConfig):
        super().__init__()

        self.attn = Attention(config)
        self.mlp = Mlp(config)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor | None = None,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        x = self.attn.forward(x, cos, sin, mask, input_pos)
        return self.mlp.forward(x)


class GIVTModel(nn.Module):
    def __init__(
        self,
        config: GIVTConfig,
    ):
        super().__init__()

        self.config = config

        self.embed = nn.Linear(
            config.input_dim, config.hidden_dim, bias=False
        )  # TODO try with bias
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim)
        self.head = nn.Linear(
            config.hidden_dim, config.input_dim * 2, bias=False
        )  # means and variances

        self.max_seq_length = config.block_size
        self.mask_cache: Tensor | None = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int):
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}"
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # overrides
        elif self.cos.device.type == "meta":
            self.cos, self.sin = self.rope_cache()
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self):
        # Trigger resetting the rope-cache
        self.max_seq_length = self.config.block_size

    def _init_weights(self, module: nn.Module):
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def rope_cache(self, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.blocks:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self):
        self.mask_cache = None
        for block in self.blocks:
            block.attn.kv_cache = None

    def forward(
        self, x: Tensor, input_pos: Tensor | None = None
    ) -> torch.distributions.Normal:
        T = x.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
            # mask = None
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.embed.forward(x)
        for block in self.blocks:
            block: Block
            x = block.forward(x, cos, sin, mask, input_pos)
        x = self.norm.forward(x)
        y = self.head.forward(x)

        y_means, y_vars = y.chunk(2, dim=-1)
        y_vars = F.softplus(y_vars) + self.config.eps
        dist = torch.distributions.Normal(y_means.float(), y_vars.float().sqrt())
        return dist


""" main """


@dataclass
class GIVTTrainingOutput:
    loss: Tensor
    info: dict | None = None


class GIVT(PreTrainedModel):
    config_class = GIVTConfig

    def __init__(self, config: GIVTConfig):
        super().__init__(config)

        self.model = GIVTModel(config)

    def forward(
        self,
        x: Tensor,
        return_info: bool = True,
    ) -> Tensor:
        dist = self.model.forward(x[:, 1:])  # (b, s, d*2)
        yhat = x[:, :-1]  # (b, s, d)

        # Compute the negative log likelihood loss
        loss = -dist.log_prob(yhat.contiguous().float()).float().mean()

        info = None
        if return_info:
            y_vars = dist.variance
            info = dict(variances=y_vars.mean().item(), means=dist.mean.mean().item())

        return GIVTTrainingOutput(loss=loss, info=info)

    def sample(
        self,
        dist: torch.distributions.Normal,
        temperature: float,
    ) -> Tensor:
        out = dist.sample()[0, [-1], :]
        return out

    def next_token(self, input_pos: Tensor, x: Tensor, **kwargs) -> Tensor:
        dist = self.model.forward(x, input_pos)
        next = self.sample(dist, **kwargs)
        return next.to(dtype=x.dtype)

    @torch.inference_mode()
    def generate(
        self,
        audio_prompt: Tensor,  # (s, d)
        max_returned_tokens: int,
        temperature: float = 0.95,
        compile: bool = False,
        show_progress: bool = True,
    ) -> Tensor:
        """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

        The implementation of this function is modified from A. Karpathy's nanoGPT.

        Args:
            audio_prompt: Tensor of shape (T) with indices of the prompt sequence.
            max_returned_tokens: The maximum number of tokens to return (given plus generated).
            temperature: Scales the predicted logits by 1 / temperature.
        """
        self.model.eval()

        self.model.max_seq_length = max_returned_tokens

        with torch.device(audio_prompt.device):
            self.model.set_kv_cache(batch_size=1)

        if compile:
            torch._dynamo.config.automatic_dynamic_shapes = True
            torch._inductor.config.triton.unique_kernel_names = True
            torch._inductor.config.coordinate_descent_tuning = True
            global next_token
            self.next_token = torch.compile(self.next_token, mode="reduce-overhead")

        T = audio_prompt.size(0)
        assert max_returned_tokens > T
        if self.model.max_seq_length < max_returned_tokens - 1:
            raise NotImplementedError(
                f"max_seq_length {self.model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
            )

        device = audio_prompt.device
        tokens = [audio_prompt]
        input_pos = torch.tensor([T], device=device)
        # prefill
        token = self.next_token(
            torch.arange(0, T, device=device),
            audio_prompt.unsqueeze(0),
            temperature=temperature,
        ).clone()
        tokens.append(token)

        pbar = tqdm(range(2, max_returned_tokens - T + 1), disable=not show_progress)
        for _ in pbar:
            token = self.next_token(
                input_pos, token.unsqueeze(0), temperature=temperature
            ).clone()
            tokens.append(token)
            input_pos = input_pos.add_(1)
        return torch.cat(tokens)
