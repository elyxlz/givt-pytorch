from givt_pytorch import GIVT, GIVTConfig, DummyDataset, Trainer, TrainConfig

model = GIVT(
    GIVTConfig(
        ema=2 / 3,
    )
)

dataset = DummyDataset()

train_config = TrainConfig(resume_from_ckpt="./logs/hwe80ykb/step_30000")


trainer = Trainer(model=model, dataset=dataset, train_config=train_config)
