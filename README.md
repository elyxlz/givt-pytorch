## GIVT-PyTorch
A partial implementation of [Generative Infinite Vocabulary Transformer (GIVT)](https://arxiv.org/abs/2312.02116) from Google Deepmind, in PyTorch.

This repo only implements the causal version of GIVT, and does away with the k mixtures predictions or the use of the full covariance matrix, as for most purposes they did not yield significantly better results.

The decoder transformer implementation is also modernized, adopting a Llama style architecture with gated MLPs, SiLU, RMSNorm, and RoPE.

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

latents = torch.randn((4, 500, 32)) # vae latents (bs, seq_len, size)
loss = model.forward(latents).loss # NLL Loss

prompt = torch.randn((50, 32)) # no batched inference implemented
generated = model.generate(
    prompt=prompt, 
    max_len=500,
    cfg_scale=0.5,
    temperature=0.95,
) # (500, 32)
```

## Training

Define a config file in `configs/`, such as this one:
```py
from givt_pytorch import (
    GIVT,
    GIVTConfig,
    DummyDataset,
    Trainer,
    TrainConfig
)

model = GIVT(GIVTConfig())
dataset = DummyDataset()

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

## TODO
- [ ] Test out with latents from an audio vae
- [ ] Add CFG with rejection sampling

## References
```bibtex
@misc{litgpt2024,
  title={lit-gpt on GitHub},
  url={https://github.com/Lightning-AI/lit-gpt},
  year={2024}

@misc{tschannen2023givt,
    title   = {GIVT: Generative Infinite-Vocabulary Transformers}, 
    author  = {Michael Tschannen, Cian Eastwood, Fabian Mentzer},
    year    = {2023},
    eprint  = {2312.02116},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```