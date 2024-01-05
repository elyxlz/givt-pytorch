from package_name import (
    DemoModel,
    DemoModelConfig,
    DemoDataset,
    Trainer,
    TrainConfig
)

model = DemoModel(DemoModelConfig())

dataset = DemoDataset()

trainer = Trainer(
    model=model,
    dataset=dataset,    
    train_config=TrainConfig()
)