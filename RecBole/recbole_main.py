import os
import pandas as pd
import argparse
import time
from recbole.quick_start import run_recbole
import wandb

# wandb.login()

def preprocess_data(data_path):
    """merged_train.csv를 읽어 RecBole 형식으로 데이터 전처리"""
    # 데이터 로드
    merged_train_df = pd.read_csv(data_path)

    # 사용자와 아이템 ID 매핑
    user2idx = {v: k for k, v in enumerate(sorted(merged_train_df['user'].unique()))}
    item2idx = {v: k for k, v in enumerate(sorted(merged_train_df['item'].unique()))}

    merged_train_df['user'] = merged_train_df['user'].map(user2idx)
    merged_train_df['item'] = merged_train_df['item'].map(item2idx)

    # Interaction 데이터 저장
    merged_train_df = merged_train_df[['user', 'item', 'time']]
    merged_train_df.columns = ['user_id:token', 'item_id:token', 'timestamp:float']

    # 데이터 저장
    output_path = "./dataset/train_data"
    os.makedirs(output_path, exist_ok=True)
    merged_train_df.to_csv(os.path.join(output_path, "train_data.inter"), sep='\t', index=False)
    print(f"Interaction data saved at {os.path.join(output_path, 'train_data.inter')}")

    return "./dataset"


def run_model(args, model_name):
    """RecBole 모델 실행"""
    config_file = 'ease_config.yaml'  # EASE 전용 설정 파일

    result = run_recbole(
        model=model_name,
        dataset='train_data',  # dataset 이름
        config_file_list=[config_file],
        config_dict={
            'data_path': args.data_path,  # 사용자로부터 전달된 data_path 사용
            **args.__dict__,
        },
    )
    return result

def main():
    # Argument 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./dataset", help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='EASE', help='Model name to run')
    args = parser.parse_args()

    # 데이터 전처리
    if not os.path.exists(os.path.join(args.data_path, "train_data.inter")):
        print("===Preprocessing data===")
        preprocess_data("./merged_train.csv")

    # 모델 실행
    print(f"Running model {args.model_name}...")
    start_time = time.time()
    result = run_model(args, args.model_name)
    elapsed_time = time.time() - start_time
    print(f"Model {args.model_name} finished in {elapsed_time / 60:.2f} minutes.")
    print(result)
    # wandb.run.finish()

if __name__ == "__main__":
    main()
