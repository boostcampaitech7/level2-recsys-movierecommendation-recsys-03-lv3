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