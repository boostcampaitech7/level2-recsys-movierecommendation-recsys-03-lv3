# src/model/MultiVAE.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=(600, 200), dropout=0.5):
        super(MultiVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1] * 2)  # mean과 logvar 생성
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], input_dim)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def forward(self, x):
        h = F.normalize(x)  # 입력 정규화
        h = self.dropout(h)
        h = self.encoder(h)
        mean, logvar = torch.chunk(h, 2, dim=-1)  # 분리
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mean
        recon_x = self.decoder(z)

        return recon_x, mean, logvar

    def loss_function_multivae(self, recon_x, x, mean, logvar, beta=1.0):
        BCE = -(x * F.log_softmax(recon_x, dim=1)).sum(dim=-1).mean()
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1).mean()

        return BCE + beta * KLD

    def init_weights(self):
        # 가중치 초기화
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, std=0.001)

        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, std=0.001)
