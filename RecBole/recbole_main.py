import argparse
import os
import time

import pandas as pd
import torch
import wandb
import yaml
from recbole.quick_start import load_data_and_model, run_recbole


def load_config(config_file: str, model_name: str) -> dict:
    """
    YAML 설정 파일에서 model_args를 불러옵니다.

    Args:
        config_file (str): 설정 파일 경로
        model_name (str): 실행할 모델 이름

    Returns:
        dict: 모델 설정을 포함한 설정 딕셔너리
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # 공통 설정 불러오기
    common_config = {key: value for key, value in config.items() if key != "model_args"}
    
    # 모델별 설정 추가
    model_config = config.get("model_args", {}).get(model_name, {})
    return {**common_config, **model_config}


def preprocess_data(data_path: str) -> str:
    """
    데이터를 RecBole 형식으로 전처리하여 저장합니다.

    Args:
        data_path (str): 원본 데이터 경로(CSV파일)

    Returns:
        str: RecBole 데이터셋 폴더 경로
    """
    # 데이터 로드
    merged_train_df = pd.read_csv(data_path)

    # 사용자와 아이템 ID 매핑
    user2idx = {v: k for k, v in enumerate(sorted(merged_train_df["user"].unique()))}
    item2idx = {v: k for k, v in enumerate(sorted(merged_train_df["item"].unique()))}

    merged_train_df["user"] = merged_train_df["user"].map(user2idx)
    merged_train_df["item"] = merged_train_df["item"].map(item2idx)
    merged_train_df["time"] = merged_train_df["time"].astype(float)
    merged_train_df["label"] = 1.0

    # Interaction 데이터 저장
    merged_train_df = merged_train_df[["user", "item", "label", "time"]]
    merged_train_df.columns = ["user_id:token", "item_id:token", "label:float", "timestamp:float"]

    # 데이터 저장
    output_path = "./dataset/train_data"
    os.makedirs(output_path, exist_ok=True)
    merged_train_df.to_csv(os.path.join(output_path, "train_data.inter"), sep="\t", index=False)
    print(f"Interaction data saved at {os.path.join(output_path, 'train_data.inter')}")

    return "./dataset"


def run_model(config: dict, model_name: str) -> dict:
    """
    RecBole 모델 실행하고 결과를 반환합니다.

    Args:
        config (dict): 모델 실행에 필요한 설정 딕셔너리
        model_name (str): 실행할 모델 이름

    Returns:
        dict: 모델 실행 결과
    """
    result = run_recbole(
        model=model_name,
        dataset="train_data",   # dataset 이름
        config_file_list=[],    # [config_file]
        config_dict=config,
    )
    return result


def generate_topk_recommendations(model_file: str, output_file: str, k: int=10) -> None:
    """
     학습된 모델을 사용해 사용자별 Top-K 추천을 직접 생성하고 CSV로 저장합니다. 

    Args:
        model_file (str): 저장된 모델 파일 경로
        output_file (str): 추천 결과를 저장할 CSV 파일 경로
        k (int, optional): 추천할 아이템 수. 기본값은 10
    """
    # 학습된 모델과 데이터 로드
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_file)

    # 전체 유저와 아이템 ID 가져오기
    user_ids = dataset.get_user_feature()[dataset.uid_field].numpy()  # 전체 사용자 ID
    item_ids = dataset.get_item_feature()[dataset.iid_field].numpy()  # 전체 아이템 ID

    # 사용자별 Top-K 추천 생성
    recommendations = []
    for user_id in user_ids:
        # 사용자가 이미 상호작용한 아이템을 가져옵니다.
        interacted_items = set(
            test_data.dataset.inter_feat[test_data.dataset.uid_field == user_id][test_data.dataset.iid_field].numpy()
        )

        # 상호작용하지 않은 아이템에 대해 점수를 예측합니다.
        scores = []
        for item_id in item_ids:
            if item_id not in interacted_items:
                input_data = {
                    dataset.uid_field: torch.tensor([user_id]),
                    dataset.iid_field: torch.tensor([item_id]),
                }
                score = model.predict(input_data)
                scores.append((item_id, score.item()))

        # 점수를 기준으로 상위 K개의 아이템을 선택합니다.
        topk_items = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

        for item, score in topk_items:
            recommendations.append({"user": user_id, "item": item, "score": score})

    # 결과를 DataFrame으로 변환
    recommendations_df = pd.DataFrame(recommendations)

    # CSV로 저장
    recommendations_df.to_csv(output_file, index=False)
    print(f"Recommendations saved to {output_file}")


def main():
    # Argument 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset", help="Path to dataset directory")
    parser.add_argument("--model_name", type=str, choices=["EASE", "LightGCN", "RecVAE", "DeepFM"], default="EASE", help="Model name to run")
    parser.add_argument("--config_file", type=str, default="./config/recbole_config.yaml", help="Path to RecBole config file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--model_path", type=str, default="EASE-Nov-27-2024_05-45-37.pth", help="Saved model path")
    args = parser.parse_args()
    
    config = load_config(args.config_file, args.model_name)
    config["epochs"] = args.epochs

    wandb.init(
        project="movie",
        name=f"{args.model_name}_test",
        entity="remember-us",
        config={
            "model": args.model_name,
            "epochs": args.epochs,
            "test_user": os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        },
    )
    # Artifacts에 폴더 내 파일 모두 업로드
    wandb.run.log_code("./")
    
    # 데이터 전처리
    if not os.path.exists(os.path.join(args.data_path, "train_data.inter")):
        print("===Preprocessing data===")
        preprocess_data("./dataset/merged_train_df.csv")
        # preprocess_data(os.path.join(args.data_path, "merged_train_df.csv"))

    # 모델 실행
    print(f"Running model {args.model_name}...")
    start_time = time.time()
    result = run_model(config, args.model_name)
    elapsed_time = time.time() - start_time

    print(f"Model {args.model_name} finished in {elapsed_time / 60:.2f} minutes.")
    print(result)

    # 모델과 데이터셋 불러오기
    print("===Loading saved model and data===")
    model_file = os.path.join("./saved/", args.model_path)  
    output_file = "/data/ephemeral/home/movie/yoon/code/output"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Saved model file not found: {model_file}")
    # 사용자별 Top-10 추천 생성 및 저장
    generate_topk_recommendations(model_file=model_file, output_file=output_file, k=10)


    wandb.log({
        "MAP@10": result["test_result"].get("map@10", 0),
        "NDCG@10": result["test_result"].get("ndcg@10", 0),
        "PRECISION@10": result["test_result"].get("precision@10", 0),
        "RECALL@10": result["test_result"].get("recall@10", 0),
        "RECALL@5": result["test_result"].get("recall@5", 0),
    })
    wandb.finish()


if __name__ == "__main__":
    main()