import torch

torch.set_float32_matmul_precision("high")

from givt_pytorch import GIVT, GIVTConfig

model: GIVT = GIVT(GIVTConfig(block_size=2000)).cuda()

latent = torch.randn(1, 50, 16).cuda()
loss = model.forward(latent).loss


for _ in range(2):
    generated = model.generate(
        latent[0], max_returned_tokens=1000, temperature=1.0, compile=True
    )

print(generated.shape)
