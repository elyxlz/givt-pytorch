import torch

from givt_pytorch import GIVT, GIVTConfig

model = GIVT(GIVTConfig(block_size=2000))

latent = torch.randn(1, 50, 16)
loss = model.forward(latent).loss

generated = model.generate(
    latent[0],
    max_returned_tokens=1000,
    temperature=1.0,
)

import pdb; pdb.set_trace()