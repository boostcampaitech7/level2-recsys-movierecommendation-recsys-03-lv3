# 데이터 경로 받아서(argparser)
# 데이터 로드한 다음에
# 모델별로 데이터셋 클래스 만들기

import torch
import pandas as pd
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    def __init__(self, args, X: pd.DataFrame, y: pd.Series):
        self.args = args
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X).shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]