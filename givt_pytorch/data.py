import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from datasets import load_dataset


class DummyDataset(Dataset):
    def __init__(
        self,
        dim: int = 16,
        seq_len: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

    def __len__(self):
        return 10000

    def __getitem__(self, idx: int):
        return dict(x=torch.randn(self.seq_len, self.dim))


class StockDataset(Dataset):
    def __init__(
        self,
        block_size: int,
        overfit: int | None = None,
    ):
        super().__init__()

        self.block_size = block_size

        # remove nones
        self.dataset: HFDataset = load_dataset("edarchimbaud/timeseries-1m-stocks")[
            "train"
        ]
        self.dataset = self.dataset.filter(lambda x: x["open"] is not None)
        self.dataset = self.dataset.remove_columns(
            [i for i in self.dataset.column_names if i != "open"]
        )

        if overfit is not None:
            self.dataset = self.dataset.select(range(overfit * block_size))

    def __len__(self):
        return len(self.dataset) // self.block_size

    def __getitem__(self, idx: int):
        item = self.dataset[idx * self.block_size : (idx + 1) * self.block_size]["open"]
        if item is None:
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        item = torch.tensor(item).unsqueeze(-1)

        return dict(x=item)
