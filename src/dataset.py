import pandas as pd
import torch
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]