import torch
from torch.utils.data import Dataset


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
