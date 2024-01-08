from givt_pytorch import GIVT, GIVTConfig, Trainer, TrainConfig
from givt_pytorch.data import StockDataset
from datasets import load_dataset, Dataset

model = GIVT(GIVTConfig(
    input_dim=1,
    # hidden_dim=256,
    # intermediate_dim=int(256*8/3),
    # num_layers=8,
    # num_heads=8,
    # block_size=2048,
))


dataset = StockDataset(block_size=4)

train_config = TrainConfig(
    name="givt-stocks",
    mixed_precision="bf16",
    ema=2/3,
    batch_size=64,
    num_workers=4,
)


trainer = Trainer(model=model, dataset=dataset, train_config=train_config)