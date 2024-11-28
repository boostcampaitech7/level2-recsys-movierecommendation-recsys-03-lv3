import os
import glob
import argparse
import torch
import pandas as pd
import numpy as np
from recbole.quick_start import load_data_and_model

def find_recent_model(model_name, saved_dir="./saved/"):
    """가장 최근에 생성된 {model_name}~.pth 파일 찾기"""
    model_files = glob.glob(os.path.join(saved_dir, f"{model_name}-*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No saved models found for model_name: {model_name} in {saved_dir}")
    # 파일 수정 시간 기준으로 정렬 후 가장 최근 파일 반환
    recent_model = max(model_files, key=os.path.getmtime)
    return recent_model

def generate_topk_recommendations(model_file, output_file, k=10):
    """ 학습된 모델을 사용해 사용자별 Top-K 추천을 생성하고 CSV로 저장하는 함수. """
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_file)

    print("Fields in Interaction object:", test_data.dataset.field2type.keys())

    user_ids = dataset.get_user_feature()[dataset.uid_field].numpy()
    item_ids = dataset.get_item_feature()[dataset.iid_field].numpy()

    print("Number of users:", len(user_ids))
    print("Number of items:", len(item_ids))

    # Interaction 데이터를 DataFrame으로 변환
    inter_feat_data = {
        field: test_data.dataset.inter_feat[field].numpy() for field in test_data.dataset.field2type.keys()
    }
    inter_feat_df = pd.DataFrame(inter_feat_data)

    # 사용자별로 이미 상호작용한 아이템 계산
    interacted_items = inter_feat_df.groupby(dataset.uid_field)[dataset.iid_field].apply(set).to_dict()

    # 사용자별 Top-K 추천 생성
    recommendations = []

    for user_id in user_ids:
        user_interacted = interacted_items.get(user_id, set())

        # 사용자별 점수 계산
        input_data = {
            dataset.uid_field: torch.tensor([user_id]).to(model.device),
            dataset.iid_field: torch.tensor(item_ids).to(model.device),
        }
        scores = model.predict(input_data).cpu().detach().numpy()

        # 상호작용하지 않은 아이템의 점수만 남기기
        filtered_scores = [
            (item, score) for item, score in zip(item_ids, scores) if item not in user_interacted
        ]

        # 상위 K개의 아이템 선택
        topk_items = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:k]

        for item, score in topk_items:
            recommendations.append({"user": user_id, "item": item})

    # 결과를 DataFrame으로 변환
    recommendations_df = pd.DataFrame(recommendations)

    # CSV로 저장
    recommendations_df.to_csv(output_file, index=False)
    print(f"Recommendations saved to {output_file}")


def main():
    # Argument 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset", help="Path to dataset directory")
    parser.add_argument("--model_name", type=str, choices=["EASE", "LightGCN"], default="EASE", help="Model name to run")
    parser.add_argument("--config_file", type=str, default="./config/recbole_config.yaml", help="Path to RecBole config file")
    parser.add_argument("--model_path", type=str, default="recent", help="Saved model path (Default = 'recent' to use the latest model)")
    args = parser.parse_args()

    # 모델과 데이터셋 불러오기
    print("===Loading saved model and data===")

    # 모델 경로 결정
    if args.model_path == "recent":
        model_file = find_recent_model(args.model_name)
        print(f"Using the most recent model: {model_file}")
    else:
        model_file = os.path.join("./saved", args.model_path)

    output_file = "../code/output/submission.csv"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Saved model file not found: {model_file}")
    # 사용자별 Top-10 추천 생성 및 저장
    generate_topk_recommendations(model_file=model_file, output_file=output_file, k=10)

if __name__ == "__main__":
    main()
