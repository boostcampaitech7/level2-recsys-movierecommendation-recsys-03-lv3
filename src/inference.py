# src/inference.py

import pandas as pd
import numpy as np
import torch
from src.preprocessing import load_data
from src.trainer import train_ease_model, train_multivae_model

def recommend_ease(model, interaction_matrix, idx_to_item, N=10):
    predictions = model.predict(interaction_matrix)
    predictions[interaction_matrix.nonzero()] = -np.inf

    top_items_per_user = []
    for user_idx in range(predictions.shape[0]):
        top_items = np.argsort(predictions[user_idx])[-N:][::-1]
        top_items_per_user.append(top_items)

    return top_items_per_user

def recommend_multivae(model, interaction_matrix, idx_to_item, device='cpu', N=10):
    model.eval()
    with torch.no_grad():
        interaction_matrix_tensor = torch.FloatTensor(interaction_matrix.toarray()).to(device)
        predictions, _, _ = model(interaction_matrix_tensor)

    predictions[interaction_matrix.nonzero()] = -np.inf
    top_items_per_user = predictions.topk(N, dim=1)[1].cpu().numpy()

    return top_items_per_user

def save_recommendations(recommendations, idx_to_user, idx_to_item, filename):
    user_ids = []
    item_ids = []
    for user_idx, items in enumerate(recommendations):
        user_id = idx_to_user[user_idx]
        for item_idx in items:
            user_ids.append(user_id)
            item_ids.append(idx_to_item[item_idx])

    output_df = pd.DataFrame({'user': user_ids, 'item': item_ids})
    output_df.to_csv(filename, index=False)
    print(f"Recommendations saved to {filename}")

def recommend_easer(model, interaction_matrix, idx_to_item, N=10):
    """
    EASER 모델을 사용한 추천 함수
    :param model: 학습된 EASER 모델
    :param interaction_matrix: 사용자-아이템 상호작용 행렬
    :param idx_to_item: 아이템 인덱스 매핑
    :param N: 추천할 아이템 개수
    :return: 사용자별 추천 아이템 인덱스
    """
    predictions = model.predict(interaction_matrix)
    predictions[interaction_matrix.nonzero()] = -np.inf

    top_items_per_user = []
    for user_idx in range(predictions.shape[0]):
        top_items = np.argsort(predictions[user_idx])[-N:][::-1]
        top_items_per_user.append(top_items)

    return top_items_per_user