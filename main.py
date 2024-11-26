# main.py

import torch
import yaml
from src.preprocessing import load_data
from src.trainer import train_ease_model, train_multivae_model, train_easer_model
from src.inference import recommend_ease, recommend_multivae, recommend_easer, save_recommendations

def main():
    # 설정 파일 로드
    with open('config/config_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 데이터 로드
    data_path = config['data_path']
    interaction_matrix, idx_to_user, idx_to_item, num_users, num_items = load_data(data_path)

    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # EASE 모델 학습 및 추천
    ease_config = config['ease']
    ease_model = train_ease_model(interaction_matrix, reg_lambda=ease_config['reg_lambda'])
    ease_recommendations = recommend_ease(ease_model, interaction_matrix, idx_to_item, N=10)
    save_recommendations(ease_recommendations, idx_to_user, idx_to_item, ease_config['output_file'])

    # MultiVAE 모델 학습 및 추천
    multivae_config = config['multivae']
    multivae_model = train_multivae_model(
        interaction_matrix,
        num_items,
        device=device,
        epochs=multivae_config['epochs'],
        batch_size=multivae_config['batch_size'],
        lr=multivae_config['lr'],
        beta=multivae_config['beta']
    )
    multivae_recommendations = recommend_multivae(multivae_model, interaction_matrix, idx_to_item, device=device, N=10)
    save_recommendations(multivae_recommendations, idx_to_user, idx_to_item, multivae_config['output_file'])

    # EASER 모델 학습 및 추천
    easer_config = config['easer']
    easer_model = train_easer_model(
        interaction_matrix,
        reg_lambda=easer_config['reg_lambda'],
        smoothing=easer_config['smoothing']
    )
    easer_recommendations = recommend_easer(easer_model, interaction_matrix, idx_to_item, N=10)
    save_recommendations(easer_recommendations, idx_to_user, idx_to_item, easer_config['output_file'])

if __name__ == "__main__":
    main()