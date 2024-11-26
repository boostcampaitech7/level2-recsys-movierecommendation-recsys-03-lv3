# src/trainer.py

import numpy as np
import torch
from torch.utils.data import DataLoader

def train_ease_model(interaction_matrix, reg_lambda=500):
    from src.model.ease import EASE
    ease_model = EASE(reg_lambda)
    ease_model.train(interaction_matrix)
    return ease_model

def train_multivae_model(interaction_matrix, num_items, device='cpu', epochs=50, batch_size=128, lr=0.001, beta=1.0):
    from src.model.multivae import MultiVAE
    multivae_model = MultiVAE(num_items).to(device)
    optimizer = torch.optim.Adam(multivae_model.parameters(), lr=lr)
    multivae_model.train()

    for epoch in range(epochs):
        total_loss = 0
        for start in range(0, interaction_matrix.shape[0], batch_size):
            end = min(start + batch_size, interaction_matrix.shape[0])
            batch = torch.FloatTensor(interaction_matrix[start:end].toarray()).to(device)

            optimizer.zero_grad()
            recon_batch, mean, logvar = multivae_model(batch)
            loss = multivae_model.loss_function(recon_batch, batch, mean, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / interaction_matrix.shape[0]:.4f}")

    return multivae_model

def train_easer_model(interaction_matrix, reg_lambda=500, smoothing=0.01):
    """
    EASER 모델 학습 함수
    :param interaction_matrix: 사용자-아이템 상호작용 행렬
    :param reg_lambda: Regularization 값
    :param smoothing: Smoothing 값
    :return: 학습된 EASER 모델
    """
    from src.model.easer import EASER
    easer_model = EASER(reg_lambda, smoothing)
    easer_model.train(interaction_matrix)
    return easer_model