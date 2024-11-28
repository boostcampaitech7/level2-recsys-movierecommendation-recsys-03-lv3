import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def ensemble_models(outputs: pd.DataFrame, p: list[int]) -> pd.DataFrame:
    """
    모델을 가중치만큼 랜덤 샘플링하여 앙상블하는 메서드

    Args:
        outputs (pd.DataFrame): 각 모델의 output을 이어붙인 데이터프레임
        p (list[int]): 각 모델의 가중치

    Returns:
        pd.DataFrame: 샘플링이 완료된 데이터프레임
    """
    output_group = outputs.groupby("user")
    sampled = pd.DataFrame(columns=["user", "item"])
    for user_id, group in tqdm(output_group, desc="Sampling..."):
        group = group.iloc[:, 1:]
        for i in range(len(p)):
            for _ in range(p[i]-1):
                group = pd.concat([group, group.iloc[:, i]], axis=1)

        items = group.values.flatten()

        lst = []
        for _ in range(10):
            r = np.random.choice(items)
            lst.append(r)
            items = items[items != r]

        result = pd.DataFrame({"user": user_id, "item": lst})
        sampled = pd.concat([sampled, result], axis=0)

    return sampled


def get_outputs(model_list: list[str], output_path: str) -> pd.DataFrame:
    """
    모델 이름을 포함한 리스트와 파일이 저장된 경로를 입력받아 ensemble_models 메서드에 입력할 수 있는 형태로 변환하는 메서드

    Args:
        model_list (list[str]): 모델의 이름을 담은 리스트
        output_path (str): csv 파일이 저장되어 있는 경로

    Returns:
        pd.DataFrame: 결과 데이터프레임
    """
    print("File Load:", *model_list)
    for i, model_name in enumerate(model_list):
        output = pd.read_csv(os.path.join(output_path, model_name) + "_output.csv")

        if i == 0:
            outputs = output
        else:
            outputs = pd.concat([outputs, output["item"]], axis=1)

    return outputs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "--m", nargs="+", type=str,
                        help="앙상블을 수행할 모델의 이름을 입력합니다. 예: --model_name EASER CatBoost MultiVAE")
    parser.add_argument("--run", "--r", type=str,
                        help="WandB run 이름을 설정할 수 있습니다.")
    parser.add_argument("--weights", "--w", nargs="+", type=int,
                        help="모델의 가중치를 입력합니다. 서로소인 정수 형태로 입력해야 합니다. 예: --weights 7 2 1")
    parser.add_argument("--output_path", "--o", type=str, default="./output/",
                        help="앙상블을 수행할 파일 경로 및 앙상블 파일의 저장 경로를 설정할 수 있습니다.")

    args = parser.parse_args()

    outputs = get_outputs(args.model_name, args.output_path)

    sampled_output = ensemble_models(outputs, args.weights)

    print(f"Sampled Output shape: {sampled_output.shape}")

    print("Saving output...")
    sampled_output.to_csv(os.path.join(args.output_path, "ensemble_output.csv"), index=False)
    print("Finished!")


if __name__ == "__main__":
    main()
