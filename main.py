# main.py

import argparse
import os
import json

import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf

import src.model as model_module
from src.utils import set_seed, check_path
from src.loader import load_data
from src.trainer import train_ease, evaluate_ease, train_multivae, evaluate_multivae
from src.inference import recommend_ease, recommend_multivae, save_recommendations


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "--m", default="DeepFM", type=str,
                        help="사용할 모델을 설정할 수 있습니다. (기본값 DeepFM)")
    parser.add_argument("--epochs", "--e", default=2, type=int,
                        help="모델 훈련을 반복할 epochs수를 지정할 수 있습니다. (기본값 2)")
    parser.add_argument("--run", "--r", type=str,
                        help="WandB run 이름을 설정할 수 있습니다.")
    parser.add_argument("--device", "--d", default="cuda", type=str,
                        choices=["cuda", "cpu"], help="device를 설정할 수 있습니다. (기본값 cuda)")

    args = parser.parse_args()

    config = "config/config_baseline.yaml"
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(config) if config else OmegaConf.create()

    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    args = config_yaml
    args_str = f"{args.model_name}_{args.run}"

    wandb.init(
        project=args.wandb_project,
        name=args_str,
        entity="remember-us",
        config={
            "model": args.model_name,
            "epochs": args.epochs,
            "test_user": os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        },
    )

    # Artifacts에 폴더 내 파일 모두 업로드
    wandb.run.log_code("./")

    set_seed(args.seed)
    check_path(args.output_path)

    print("--------------- LOAD DATA ---------------")
    # 디바이스 설정
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # 데이터 로드
    data_path = args.data_path
    interaction_matrix, train_data, test_data, idx_to_user, idx_to_item, _, num_items = load_data(data_path)

    print(f"--------------- INIT {args.model_name} ---------------")
    if args.model_name == "MultiVAE":
        args.model_args.MultiVAE.input_dim = num_items
        model = getattr(model_module, args.model_name)(**args.model_args[args.model_name]).to(device)
    else:
        model = getattr(model_module, args.model_name)(**args.model_args[args.model_name])

    print(f"--------------- {args.model_name} TRAINING ---------------")
    match args.model_name:
        case "EASE":
            # EASE 모델 학습 및 평가
            model = train_ease(model, interaction_matrix)
            train_loss, n10, r10 = evaluate_ease(model, train_data, test_data, top_k=10)
            print("=" * 77)
            print(f"| End of training | train loss {train_loss:5.4f} | ndgc@10 {n10:5.4f} | recall@10 {r10:5.4f} |")
            print("=" * 77)
        
        case "EASER":
            # EASER 모델 학습 및 평가
            model = train_ease(model, interaction_matrix)
            train_loss, n10, r10 = evaluate_ease(model, train_data, test_data, top_k=10)
            print("=" * 77)
            print(f"| End of training | train loss {train_loss:5.4f} | ndgc@10 {n10:5.4f} | recall@10 {r10:5.4f} |")
            print("=" * 77)
        
        case "MultiVAE":
            # MultiVAE 모델 학습 및 평가
            model = train_multivae(
                model,
                interaction_matrix,
                epochs=args.epochs,
                batch_size=512,
                lr=0.001,
                beta=1.0,
                device=args.device
            )
            train_loss, n10, r10 = evaluate_multivae(
                model, 
                train_data,
                test_data,
                batch_size=512,
                beta=1.0,
                device=args.device
            )
            print("=" * 77)
            print("| End of training | train loss {:5.4f} | ndgc@10 {:5.4f} | recall@10 {:5.4f} |".format(train_loss, n10, r10))
            print("=" * 77)
            

    print(f"--------------- {args.model_name} TEST ---------------")
    match args.model_name:
        case "EASE":
            ease_recommendations = recommend_ease(model, interaction_matrix, idx_to_item, N=10)
            save_recommendations(ease_recommendations, idx_to_user, idx_to_item, filename=args.model_name, output_path=args.output_path)
        
        case "EASER":
            easer_recommendations = recommend_ease(model, interaction_matrix, idx_to_item, N=10)
            save_recommendations(easer_recommendations, idx_to_user, idx_to_item, filename=args.model_name, output_path=args.output_path)
        
        case "MultiVAE":
            # model = train_multivae(model, interaction_matrix, epochs=args.epochs, batch_size=512, lr=0.001, beta=1.0, device=args.device)
            model.eval()
            with torch.no_grad():
                interaction_tensor = torch.FloatTensor(interaction_matrix.toarray()).to(device)
                predictions, _, _ = model(interaction_tensor)
            
            # 평가된 아이템 제외
            predictions[interaction_matrix.nonzero()] = -np.inf

            # 추천 결과 파일 저장
            multivae_recommendations = recommend_multivae(model, interaction_matrix, idx_to_item, device=device, N=10)
            save_recommendations(multivae_recommendations, idx_to_user, idx_to_item, filename=args.model_name, output_path=args.output_path)

    
    wandb.log({
        "features": "user-item interaction matrix",
        "params": json.dumps(OmegaConf.to_container(args.model_args[args.model_name]))
    })
    wandb.finish()


if __name__ == "__main__":
    main()