A partial implementation of Generative Infinite Vocabulary Transformer (GIVT) from Google Deepmind, in PyTorch.

## Install
```sh
# for training/development
pip install -e '.[train]'

# for inference
pip install .
```


## Usage (inference)
```py
from givt_pytorch import GIVT

# load pretrained checkpoint
model = GIVT.from_pretrained(xxx)
```

## Usage (training)

Define a config file in `configs/`, called `demo_run` in this case:
```py
from package_name import (
    GIVT,
    GIVTCONfig,
    DemoDataset,
    Trainer,
    TrainConfig
)

model = GIVT(GIVTConfig(xxx))
dataset = DemoDataset(xxx)

trainer = Trainer(
    model=model,
    dataset=dataset,    
    train_config=TrainConfig(xxx)
)
```
Create an accelerate config.
```sh
accelerate config
```

And then run the training.
```sh
accelerate launch train.py demo_run
```
