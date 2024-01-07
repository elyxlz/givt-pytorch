from givt_pytorch import GIVT, GIVTConfig, DummyDataset, Trainer, TrainConfig

model = GIVT(GIVTConfig())

dataset = DummyDataset()

trainer = Trainer(model=model, dataset=dataset, train_config=TrainConfig())
