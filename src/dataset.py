import pandas as pd
import torch
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.long()
        self.y = y.long()
    
    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, index: int):
        if self.y is None:
            return self.X[index]
        else:
            return self.X[index], self.y[index]
