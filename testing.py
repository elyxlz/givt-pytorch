import torch

from givt_pytorch import GIVT, GIVTConfig

model = GIVT(GIVTConfig())

latent = torch.randn(1, 50, 16)
loss = model.forward(latent).loss

import pdb

pdb.set_trace()
