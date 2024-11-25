import pandas as pd
import torch
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int):
        if self.y is None:
            return self.X[index]
        else:
            return self.X[index], self.y[index]