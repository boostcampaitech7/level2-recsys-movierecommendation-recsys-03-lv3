import os
import json
import argparse

import pandas as pd
import numpy as np
import wandb
from omegaconf import OmegaConf

import src.model as model_module
import src.trainer as trainer_module
from src.loader import load_dataset, data_split, data_loader
from src.utils import set_seed, check_path, EarlyStopping
from src.preprocessing import replace_id


def main():

    # 1. Basic Environment Setup

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
    checkpoint = args_str + ".pt"
    checkpoint_path = os.path.join(args.output_path, checkpoint)

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

    # merged_train_df = load_dataset(args)
    merged_train_data = os.path.join(args.output_path, "merged_train_df.csv")
    # merged_train_df.to_csv(merged_train_data, index=False)
    merged_train_df = pd.read_csv(merged_train_data)

    merged_train_df, users_dict, items_dict = replace_id(merged_train_df)
    items_dict = {k: v + len(users_dict) for k, v in items_dict.items()}
    merged_train_df.drop(columns=["title", "genre", "director", "time", "year", "num_reviews_item"], inplace=True)
    wandb.log({"features": list(merged_train_df.columns)})

    X_train, X_valid, y_train, y_valid = data_split(args, merged_train_df)

    seen_indices = y_train[y_train == 1].index
    seen_data = X_train.loc[seen_indices]
    seen_items = seen_data.groupby("user")["item"].apply(list).to_dict()

    train_loader = data_loader(["user", "item"], 1024, X_train, y_train, True)
    valid_loader = data_loader(["user", "item"], 512, X_valid, y_valid, True)

    print("--------------- INIT {args.model_name} ---------------")
    args.model_args[args.model_name].input_dims = [len(users_dict), len(items_dict)]
    model = getattr(model_module, args.model_name)(**args.model_args[args.model_name]).to(args.device)

    print("--------------- {args.model_name} TRAINING ---------------")

    trainer = getattr(trainer_module, args.model_name)(model, train_loader, valid_loader, None, seen_items, args)

    early_stopping = EarlyStopping(checkpoint_path, patience=10, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)
        scores = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("--------------- {args.model_name} TEST ---------------")
    trainer.load(checkpoint_path)
    _ = trainer.test(0)

    wandb.log({
        "params": json.dumps(OmegaConf.to_container(args.model_args[args.model_name]))
    })
    wandb.finish()


if __name__ == "__main__":
    main()
