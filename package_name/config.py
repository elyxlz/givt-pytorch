from transformers import PretrainedConfig

class DemoModelConfig(PretrainedConfig):
    model_type = "demo_model"

    def __init__(
        self,
        size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
