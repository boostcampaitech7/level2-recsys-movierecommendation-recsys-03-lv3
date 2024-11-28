# src/model/MultiVAE.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix


class MultiVAE(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            hidden_dims: tuple[int, int] =(600, 200), 
            dropout: float =0.5
        ):
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MultiVAE 모델의 순전파 메서드

        Args:
            x (torch.Tensor): 입력 데이터 (user-item interaction matrix)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 재구성된 데이터 및 latent space의 평균, 로그 분산
        """
        h = F.normalize(x)  # 입력 정규화
        h = self.dropout(h)
        h = self.encoder(h)
        mean, logvar = torch.chunk(h, 2, dim=-1)  # 분리
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mean
        recon_x = self.decoder(z)

        return recon_x, mean, logvar

    def loss_function_multivae(
            self, 
            x: torch.Tensor,
            recon_x: torch.Tensor,
            mean: torch.Tensor, 
            logvar: torch.Tensor, 
            beta: float =1.0
        ) -> float:
        """
        MultiVAE 모델의 loss를 계산하는 메서드

        Args:
            x (torch.Tensor): 입력 데이터
            recon_x (torch.Tensor): 재구성된 데이터
            mean (torch.Tensor): latent space의 평균
            logvar (torch.Tensor): latent space의 로그 분산
            beta (float, optional): KL diverence에 대한 가중치. 기본값은 1.0

        Returns:
            float: 계산된 loss 값
        """
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
