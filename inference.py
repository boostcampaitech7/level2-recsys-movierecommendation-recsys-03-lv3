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
    merged_train_df.drop(columns=["title", "genre", "director", "time", "year", "num_reviews_item"], inplace=True)

    input_dims = merged_train_df.drop(columns=["review"]).nunique().tolist()
    args.model_args[args.model_name].input_dims = input_dims

    X_train, y_train = merged_train_df.drop(columns=["review"]), merged_train_df["review"]

    per_users_df = merged_train_df.groupby("user", group_keys=False).apply(lambda x: list(zip(x["item"].values, x["review"].values)), include_groups=False)
    user_groups = []
    for user, item_review in per_users_df.items():
        items, reviews = zip(*item_review)
        user_seen_items = [item for item, review in zip(items, reviews) if review == 1]
        user_groups.append((user, list(items), list(reviews), user_seen_items))

    submission_loader = data_loader(["user", "item"], 1024, X_train, y_train, False)

    print(f"--------------- INIT {args.model_name} ---------------")
    model = getattr(model_module, args.model_name)(**args.model_args[args.model_name]).to(args.device)

    print(f"--------------- PREDICT {args.model_name} ---------------")

    trainer = getattr(trainer_module, args.model_name)(model, None, None, submission_loader, user_groups, args)

    trainer.load(checkpoint_path)
    preds = trainer.submission(0)

    generate_submission_file(data_path, preds, items_dict)


if __name__ == "__main__":
    main()
