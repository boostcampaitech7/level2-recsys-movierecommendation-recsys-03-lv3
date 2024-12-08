# src/model/DeepFM.py

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(
            self,
            input_dims: list[int],
            embedding_dim: int,
            mlp_dims: list[int],
            drop_rate: float = 0.1
        ) -> None:
        """
        Args:
            input_dims (list[int]): 각 입력 차원의 크기 리스트 (예: 사용자 수, 영화 수, 장르 수)
            embedding_dim (int): 임베딩 차원
            mlp_dims (list[int]): MLP의 각 레이어 차원 리스트
            drop_rate (float, optional): 드롭아웃 비율 (기본값 0.1)
        """
        super(DeepFM, self).__init__()
        total_input_dim = int(sum(input_dims))  # n_user + n_movie + n_genre

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1)

        self.embedding = nn.Embedding(total_input_dim, embedding_dim)
        self.embedding_dim = len(input_dims) * embedding_dim

        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i == 0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i - 1], dim))
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))

        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Factorization Machine(FM)을 통해 예측 값을 계산하는 메서드

        Args:
            x (torch.Tensor): 입력 텐서 (batch_size, total_num_input)

        Returns:
            torch.Tensor: FM의 출력 값
        """
        # x : (batch_size, total_num_input)
        embed_x = self.embedding(x)

        fm_y = self.bias + torch.sum(self.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return fm_y

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLP를 통해 예측 값을 계산하는 메서드

        Args:
            x (torch.Tensor): 입력 텐서 (batch_size, total_num_input)

        Returns:
            torch.Tensor: MLP의 출력 값
        """
        embed_x = self.embedding(x)

        inputs = embed_x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)

        return mlp_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파 메서드

        Args:
            x (torch.Tensor): 입력 텐서 (batch_size, total_num_input)

        Returns:
            torch.Tensor: 최종 예측 값 (sigmoid 함수 적용)
        """
        # fm component
        fm_y = self.fm(x).squeeze(1)

        # deep component
        mlp_y = self.mlp(x).squeeze(1)

        y = torch.sigmoid(fm_y + mlp_y)

        return y
