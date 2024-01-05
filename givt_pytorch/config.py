from transformers import PretrainedConfig

class GIVTConfig(PretrainedConfig):
    model_type = "givt"

    def __init__(
        self,
        size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
