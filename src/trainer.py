# src/trainer.py

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix

from src.model.ease import EASE
from src.model.easer import EASER
from src.model.multivae import MultiVAE
from src.utils import ndcg_binary_at_k_batch, recall_at_k_batch


def train_ease(model: object, data: csr_matrix) -> object:
    """
    EASE 계열 모델(EASE, EASER)을 학습하는 함수

    Args:
        model (object): EASER 모델 객체
        data (csr_matrix): 사용자-아이템 상호작용 희소 행렬

    Returns:
        object: 학습된 EASE 계열 모델 객체
    """
    model.train(data)
    return model


def evaluate_ease(model: object, data: csr_matrix) -> np.ndarray:
    """
    학습된 EASE 계열 모델(EASE, EASER)을 이용해 예측값을 반환하는 함수

    Args:
        model (object): 학습된 EASE 계열 모델 객체
        data (csr_matrix): 사용자-아이템 상호작용 희소 행렬

    Returns:
        np.ndarray: 추천 점수 행렬
    """
    pred = model.predict(data)
    return pred


def train_multivae(
        model, 
        train_data,  
        epochs, 
        batch_size, 
        lr, 
        beta,
        device="cuda"
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
        wandb.log({"loss": total_loss / train_data.shape[0]})
    return model


def evaluate_multivae(
        model, 
        train_data, 
        valid_data, 
        batch_size, 
        beta=1.0,
        device="cuda"
    ):
    model.eval()
    
    # loss와 평가 지표를 담을 리스트 생성
    total_valid_loss_list = []
    n10_list = []
    r10_list = []

    with torch.no_grad():
        for start in range(0, train_data.shape[0], batch_size):
            end = min(start + batch_size, train_data.shape[0])    
            batch = torch.FloatTensor(train_data[start:end].toarray()).to(device)
            heldout_batch = valid_data[start:end]

            recon_batch, mean, logvar = model(batch)
            loss = model.loss_function(recon_batch, batch, mean, logvar, beta)
            total_valid_loss_list.append(loss.item())

            # 평가된 아이템 제외
            recon_batch = recon_batch.cpu().numpy()
            batch = batch.cpu().numpy()
            recon_batch[batch.nonzero()] = -np.inf
            
            # NDCG@10, Recall@10 계산
            n10 = ndcg_binary_at_k_batch(recon_batch, heldout_batch, 10)
            r10 = recall_at_k_batch(recon_batch, heldout_batch, 10)

            n10_list.append(n10)
            r10_list.append(r10)

    n10_list = np.concatenate(n10_list)
    r10_list = np.concatenate(r10_list)

    post_fix = {
            "RECALL@10": "{:.4f}".format(np.nanmean(r10_list)),
            "NDCG@10": "{:.4f}".format(np.nanmean(n10_list)),
    }
    wandb.log(post_fix)

    return np.nanmean(total_valid_loss_list), np.nanmean(n10_list), np.nanmean(r10_list)