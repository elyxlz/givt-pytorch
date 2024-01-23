from givt_pytorch import GIVT, GIVTConfig, Trainer, TrainConfig
from givt_pytorch.data import StockDataset

model = GIVT(
    GIVTConfig(
        input_dim=1,
        hidden_dim=1536,
        intermediate_dim=int(1536 * 8 / 3),
        num_layers=16,
        num_heads=8,
        block_size=60,
    )
)


dataset = StockDataset(block_size=60, overfit=1)
print(len(dataset))

train_config = TrainConfig(
    name="givt-stocks-v1",
    mixed_precision="bf16",
    ema=2/3,
    batch_size=128,
    num_workers=8,
    log_every=30,
    val_every=500,
    use_wandb=True,
    push_every=10000,
    fused_adam=True,
)


trainer = Trainer(model=model, dataset=dataset, train_config=train_config)
