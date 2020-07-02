# override __init__ __getitem__ __len__
import torch
from torch.utils import data


class TwitterDataset(data.Dataset):
    """
    we need a shape like [data_num, data_len]
    input shape is [data_num, seq_len, feature_dim]
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
    