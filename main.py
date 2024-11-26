# main.py

import torch
from src.preprocessing import load_data
from src.trainer import train_ease_model, train_multivae_model
from src.inference import recommend_ease, recommend_multivae, save_recommendations

def main():
    # 데이터 로드
    data_path = "/data/ephemeral/home/movie/data/train/train_ratings.csv"
    interaction_matrix, idx_to_user, idx_to_item, num_users, num_items = load_data(data_path)

    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # EASE 모델 학습 및 추천
    ease_model = train_ease_model(interaction_matrix, reg_lambda=500)
    ease_recommendations = recommend_ease(ease_model, interaction_matrix, idx_to_item, N=10)
    save_recommendations(ease_recommendations, idx_to_user, idx_to_item, 'ease.csv')

    # MultiVAE 모델 학습 및 추천
    multivae_model = train_multivae_model(interaction_matrix, num_items, device=device)
    multivae_recommendations = recommend_multivae(multivae_model, interaction_matrix, idx_to_item, device=device, N=10)
    save_recommendations(multivae_recommendations, idx_to_user, idx_to_item, 'multivae.csv')

if __name__ == "__main__":
    main()
