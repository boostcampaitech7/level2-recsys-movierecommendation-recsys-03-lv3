# src/trainer.py

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.ease import EASE
from src.model.easer import EASER
from src.model.multivae import MultiVAE
from src.utils import ndcg_binary_at_k_batch, recall_at_k_batch


def train_ease_model(model, interaction_matrix, reg_lambda=500):
    model.train(interaction_matrix)
    return model


def train_multivae_model(
        model, 
        train_data,  
        epochs, 
        batch_size, 
        lr, 
        beta,
        device
    ):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for start in range(0, train_data.shape[0], batch_size):
            end = min(start + batch_size, train_data.shape[0])
            batch = torch.FloatTensor(train_data[start:end].toarray()).to(device)

            optimizer.zero_grad()
            recon_batch, mean, logvar = model(batch)
            loss = model.loss_function(recon_batch, batch, mean, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / train_data.shape[0]:.4f}")

    return model


def evaluate_multivae_model(
        model, 
        train_data, 
        valid_data, 
        batch_size, 
        beta=1.0,
        device
    ):
    model.eval()
    
    # loss와 평가 지표를 담을 리스트 생성
    total_valid_loss_list = []
    n10_list = []
    r10_list = []

    with torch.no_grad():
        for start in range(0, train_data.shape[0], batch_size):
            end = min(start + batch_size, train_data.shape[0], train_data.shape[0])    
            batch = torch.FloatTensor(train_data[start:end].toarray()).to(device)
            heldout_batch = valid_data[start:end]
        recon_batch, mean, logvar = model(batch)
        loss = model.loss_function(recon_batch, batch, mean, logvar, beta)

        total_valid_loss_list.append(loss.item())
        # 평가된 아이템 제외
        recon_batch = recon_batch.cpu().numpy()
        recon_batch[train_data.nonzero()] = -np.inf

        n10 = ndcg_binary_at_k_batch(recon_batch, heldout_batch, 10)
        r10 = recall_at_k_batch(recon_batch, heldout_batch, 10)

        n10_list.append(n10)
        r10_list.append(r10)

    n10_list = np.concatenate(n10_list)
    r10_list = np.concatenate(r10_list)

    return np.nanmean(total_valid_loss_list), np.nanmean(n10_list), np.nanmean(r10_list)


def train_easer_model(model, interaction_matrix, reg_lambda=500, smoothing=0.01):
    """
    EASER 모델 학습 함수
    :param interaction_matrix: 사용자-아이템 상호작용 행렬
    :param reg_lambda: Regularization 값
    :param smoothing: Smoothing 값
    :return: 학습된 EASER 모델
    """
    model.train(interaction_matrix)
    return model