## GIVT-PyTorch
A partial implementation of Generative Infinite Vocabulary Transformer (GIVT) from Google Deepmind, in PyTorch.

This repo only implements the causal version of GIVT, and does away with the k mixtures predictions or the use of the full covariance matrix, as for most purposes they did not yield better results. The end-goal of this repo is to apply GIVT as a simple and powerful solution to audio generation with Audio VAEs.

The decoder transformer implementation is also modernized, adopting a Llama style architecture with gated MLPs, SilU, RMSNorm, and rotary positional embeddings.

## Install
```sh
# for inference
pip install .

# for training/development
pip install -e '.[train]'
```


## Usage
```py
from givt_pytorch import GIVT

# load pretrained checkpoint
model = GIVT.from_pretrained('elyxlz/givt-test')

latents = torch.randn((4, 500, 32)) # audio vae latents (bs, seq_len, size)
loss = model.forward(latents) # NLL Loss

prompt = torch.randn((4, 50, 32))
generated = model.generate(
    prompt=prompt,
    max_len=500,
    temperature=0.95,
    cfg_scale=0.5,
    temperature=0.95,
)
```

## Training

Define a config file in `configs/`, such as this one:
```py
from package_name import (
    GIVT,
    GIVTConfig,
    DemoDataset,
    Trainer,
    TrainConfig
)

model = GIVT(GIVTConfig())
dataset = DemoDataset()

trainer = Trainer(
    model=model,
    dataset=dataset,    
    train_config=TrainConfig()
)
```
Create an accelerate config.
```sh
accelerate config
```

And then run the training.
```sh
accelerate launch train.py {config_name}
```
