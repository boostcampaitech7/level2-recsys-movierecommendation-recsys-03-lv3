# src/inference.py

from argparse import Namespace
import pandas as pd
import numpy as np
import torch


def recommend_ease(
        model: object, 
        interaction_matrix: np.ndarray, 
        idx_to_item, 
        N: int =10
    ) -> list[list[float]]:
    """
    예측값에서 유저별 N개의 추천 아이템을 추려내는 함수

    Args:
        model (object): 학습된 EASE 계열 모델 객체
        interaction_matrix (np.ndarray): 예측 점수 행렬
        idx_to_item (_type_): 
        N (int, optional): 추천할 아이템 개수. 디폴트는 10

    Returns:
        list[list[float]]: 유저별 N개의 추천 아이템 리스트
    """
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


def save_recommendations(
        recommendations: list[list[float]], 
        idx_to_user: dict[int, int], 
        idx_to_item: dict[int, int], 
        filename: str,
        args
    ) -> None:
    """
    추천 결과를 submission을 위한 양식에 맞게 바꾼 후, 파일로 저장하는 함수

    Args:
        recommendations (list[list[float]]): 유저별 추천 아이템 리스트
        idx_to_user (dict[int, int]): 인덱스를 유저 ID로 매핑시키기 위한 딕셔너리
        idx_to_item (dict[int, int]): 인덱스를 아이템 ID로 매핑시키기 위한 딕셔너리
        filename (str): 저장할 파일 이름
    
    Return:
        None
    """
    user_ids = []
    item_ids = []
    for user_idx, items in enumerate(recommendations):
        user_id = idx_to_user[user_idx]
        for item_idx in items:
            user_ids.append(user_id)
            item_ids.append(idx_to_item[item_idx])

    output_df = pd.DataFrame({'user': user_ids, 'item': item_ids})
    output_df.to_csv(f"{args.output_path}{filename}.csv", index=False)
    print(f"Recommendations saved to {filename}")