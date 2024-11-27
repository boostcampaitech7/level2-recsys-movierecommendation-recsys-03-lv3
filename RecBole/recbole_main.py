import os
import pandas as pd
import argparse
import time
import yaml
#import glob
from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils.case_study import full_sort_topk
import wandb

# wandb.login()

def load_config(config_file, model_name):
    """YAML 설정 파일에서 model_args를 불러오기"""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # 공통 설정 불러오기
    common_config = {key: value for key, value in config.items() if key != "model_args"}
    
    # 모델별 설정 추가
    model_config = config.get("model_args", {}).get(model_name, {})
    return {**common_config, **model_config}

def preprocess_data(data_path):
    """merged_train.csv를 읽어 RecBole 형식으로 데이터 전처리"""
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


def run_model(config, model_name):
    """RecBole 모델 실행"""
    # config_file = f"{model_name.lower()}_config.yaml" # 모델별 config파일 생성 시
    result = run_recbole(
        model=model_name,
        dataset="train_data",  # dataset 이름
        config_file_list=[], # [config_file]
        config_dict=config,
    )
    return result

# def find_saved_model(model_name):
#     """저장된 모델 파일을 찾는 함수"""
#     saved_files = glob.glob(f"./saved/{model_name}-*.pth")  # 모델 이름과 패턴에 맞는 파일 탐색
#     if saved_files:
#         # 가장 최근 파일 선택 (필요하면 정렬 기준 추가 가능)
#         return saved_files[-1]
#     else:
#         raise FileNotFoundError(f"No saved model found for model: {model_name}")

def generate_topk_recommendations(model_file, output_file, topk=10):
    """ 학습된 모델을 사용해 사용자별 Top-K 추천을 생성하고 CSV로 저장하는 함수. """
    # 학습된 모델과 데이터 로드
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_file)

    # history_item 값 확인
    print(test_data.dataset.inter_feat["history_item"])

    # 전체 유저와 아이템 ID 가져오기
    user_ids = dataset.get_user_feature()[dataset.uid_field].numpy()  # 전체 사용자 ID
    item_ids = dataset.get_item_feature()[dataset.iid_field].numpy()  # 전체 아이템 ID

    # 사용자별 Top-K 추천 생성
    topk_items = full_sort_topk(
        uid_series=user_ids,
        model=model,
        test_data=test_data,
        k=topk,
    )

    # 결과를 DataFrame으로 변환
    recommendations = []
    for user_id, items in zip(user_ids, topk_items.cpu().numpy()):
        for item in items:
            recommendations.append({"user": user_id, "item": item})
    
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
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--model_path", type=str, default="EASE-Nov-26-2024_13-42-38.pth", help="Saved model path")
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
    generate_topk_recommendations(model_file=model_file, output_file=output_file, topk=10)


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