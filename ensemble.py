# src/ensemble.py

import argparse
import os

import numpy as np
import pandas as pd

from src.utils import get_outputs, ensemble_models


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "--m", nargs="+", type=str,
                        help="앙상블을 수행할 모델의 이름을 입력합니다. 예: --model_name EASER CatBoost MultiVAE")
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
