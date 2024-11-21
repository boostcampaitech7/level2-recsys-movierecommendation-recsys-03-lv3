import pandas as pd
import torch
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    def __init__(self, args, X: pd.DataFrame, y: pd.Series):
        self.args = args
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X).shape[0]

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]