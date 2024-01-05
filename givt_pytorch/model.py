import functools
import pdb

from torch import nn, Tensor
from transformers import PreTrainedModel

from .config import GIVTConfig

""" debug """

def debug_func(func):
    """ Decorator to debug a function when error is encountered."""
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


""" Build up your model here"""

class GIVT(PreTrainedModel):
    config_class = GIVTConfig

    def __init__(self, config: GIVTConfig):
        super().__init__(config)

        self.net = nn.Linear(config.size, config.size)

    def forward(self, x: Tensor) -> Tensor:
        loss = 42 - self.net.forward(x)
        return loss

    def generate(self, x: Tensor) -> Tensor:
        return self.net.forward(x)
