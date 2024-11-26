import os
import argparse

import pandas as pd
from omegaconf import OmegaConf

import src.model as model_module
import src.trainer as trainer_module
from src.utils import set_seed, check_path, generate_submission_file
from src.loader import load_dataset, data_loader
from src.preprocessing import replace_id


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "--m", default="DeepFM", type=str,
                        help="사용할 모델을 설정할 수 있습니다. (기본값 DeepFM)")
    parser.add_argument("--run", "--r", type=str,
                        help="불러올 .pt 파일의 모델명 뒤 run명을 설정할 수 있습니다.")
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
    data_path = os.path.join(args.data_path, "train_ratings.csv")

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

    X_train, y_train = merged_train_df.drop(columns=["review"]), merged_train_df["review"]

    seen_data = X_train.loc[y_train[y_train == 1].index]
    seen_data["item"] = seen_data["item"] + len(users_dict)
    seen_items = seen_data.groupby("user")["item"].apply(list).to_dict()

    submission_loader = data_loader(["user", "item"], 1024, X_train, y_train, False)

    print(f"--------------- INIT {args.model_name} ---------------")
    args.model_args[args.model_name].input_dims = [len(users_dict), len(items_dict)]
    model = getattr(model_module, args.model_name)(**args.model_args[args.model_name]).to(args.device)

    print(f"--------------- PREDICT {args.model_name} ---------------")

    trainer = getattr(trainer_module, args.model_name)(model, None, None, submission_loader, seen_items, args)

    trainer.load(checkpoint_path)
    preds = trainer.submission(0)

    generate_submission_file(data_path, preds, items_dict)


if __name__ == "__main__":
    main()
