A minimalistic and hackable template for developing, training, and sharing deep learning models.


## Stack
- Pytorch
- Accelerate
- Huggingface hub
- Wandb


## Install
```sh
# for training/development
pip install -e '.[train]'

# for inference
pip install .
```

## Structure
```
├── package_name
│   ├── config.py # model config
│   ├── data.py # data processing logic
│   ├── model.py # model definition
│   └── trainer.py # trainer class and train config
```

## Usage (inference)
```py
from package_name import DemoModel

# load pretrained checkpoint
model = DemoModel.from_pretrained(xxx)
```

## Usage (training)

Define a config file in `configs/`, called `demo_run` in this case:
```py
from package_name import (
    DemoModel,
    DemoModelConfig,
    DemoDataset,
    Trainer,
    TrainConfig
)

model = DemoModel(DemoModelConfig(xxx))
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
