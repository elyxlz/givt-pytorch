from transformers import PretrainedConfig


class GIVTConfig(PretrainedConfig):
    model_type = "givt"

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 32,
        intermediate_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        block_size: int = 512,
        eps: float = 1e-8,
        rope_base: int = 1000000,
        rope_rotary_percentage: float = 1.0,
        rope_condense_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.block_size = block_size
        self.eps = eps
        self.rope_base = rope_base
        self.rope_condense_ratio = rope_condense_ratio
        self.rope_rotary_percentage = rope_rotary_percentage

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        self.rope_n_elem = int(self.rope_rotary_percentage * self.head_dim)
