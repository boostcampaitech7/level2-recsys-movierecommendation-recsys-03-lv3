import os

import argparse
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, random_split
from scipy.sparse import csr_matrix

from src.utils import set_seed, check_path, generate_submission_file
import src.model as model_module
import src.trainer as trainer_module


def main():

    # 1. Basic Environment Setup
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
    data_file = args.data_path + "train_ratings.csv"

    set_seed(args.seed)
    check_path(args.output_path)

    # 2. Load Data
    # 1. Rating df 생성
    rating_data = args.data_path + "train_ratings.csv"

    raw_rating_df = pd.read_csv(rating_data)
    raw_rating_df
    raw_rating_df["rating"] = 1.0 # implicit feedback
    raw_rating_df.drop(["time"],axis=1,inplace=True)

    users = set(raw_rating_df.loc[:, "user"])
    items = set(raw_rating_df.loc[:, "item"])

    #2. Genre df 생성
    genre_data = args.data_path + "/genres.tsv"

    raw_genre_df = pd.read_csv(genre_data, sep="\t")
    raw_genre_df = raw_genre_df.drop_duplicates(subset=["item"]) #item별 하나의 장르만 남도록 drop

    genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df["genre"]))}
    raw_genre_df["genre"]  = raw_genre_df["genre"].map(lambda x : genre_dict[x]) #genre id로 변경

    # 3. Negative instance 생성
    # print("Create Nagetive instances")
    # num_negative = 50
    # user_group_dfs = list(raw_rating_df.groupby("user")["item"])
    # first_row = True
    # user_neg_dfs = pd.DataFrame()

    # for u, u_items in user_group_dfs:
    #     u_items = set(u_items)
    #     i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)

    #     i_user_neg_df = pd.DataFrame({"user": [u]*num_negative, "item": i_user_neg_item, "rating": [0]*num_negative})
    #     if first_row == True:
    #         user_neg_dfs = i_user_neg_df
    #         first_row = False
    #     else:
    #         user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)

    # raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis = 0, sort=False)

    # 4. Join dfs
    # joined_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on="item", right_on="item", how="inner")

    # joined_rating_df.to_csv(os.path.join("output/", "joined_rating_df.csv"), index=False)
    joined_rating_df = pd.read_csv("output/joined_rating_df.csv")

    users = list(set(joined_rating_df.loc[:,"user"]))
    users.sort()
    items =  list(set((joined_rating_df.loc[:, "item"])))
    items.sort()
    genres =  list(set((joined_rating_df.loc[:, "genre"])))
    genres.sort()

    users_dict = {users[i]: i for i in range(len(users))}
    joined_rating_df["user"]  = joined_rating_df["user"].map(lambda x : users_dict[x])
    users = list(set(joined_rating_df.loc[:,"user"]))

    items_dict = {items[i]: i for i in range(len(items))}
    joined_rating_df["item"]  = joined_rating_df["item"].map(lambda x : items_dict[x])
    items =  list(set((joined_rating_df.loc[:, "item"])))

    joined_rating_df = joined_rating_df.sort_values(by=["user"])
    joined_rating_df.reset_index(drop=True, inplace=True)

    data = joined_rating_df
    users = list(set(joined_rating_df.loc[:,"user"]))
    items =  list(set((joined_rating_df.loc[:, "item"])))
    genres =  list(set((joined_rating_df.loc[:, "genre"])))

    n_user = len(users)
    n_item = len(items)
    n_genre = len(genres)

    #6. feature matrix X, label tensor y 생성
    class RatingDataset(Dataset):
        def __init__(self, input_tensor, target_tensor):
            self.input_tensor = input_tensor.long()
            self.target_tensor = target_tensor.long()

        def __getitem__(self, index):
            return self.input_tensor[index], self.target_tensor[index]

        def __len__(self):
            return self.target_tensor.size(0)

    def df_to_dataloader(data, batch_size, shuffle):
        user_col = torch.tensor(data.loc[:,"user"])
        item_col = torch.tensor(data.loc[:,"item"])
        genre_col = torch.tensor(data.loc[:,"genre"])

        users = list(set(data.loc[:,"user"]))
        items =  list(set((data.loc[:, "item"])))
        n_user = len(users)
        n_item = len(items)

        offsets = [0, n_user, n_user + n_item]
        for col, offset in zip([user_col, item_col, genre_col], offsets):
            col += offset

        X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1)
        y = torch.tensor(list(data.loc[:,"rating"]))

        dataset = RatingDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader
    
    submission_loader = df_to_dataloader(data, 1024, False)

    def df_to_matrix(df):
        user_item_dict = {}
        
        df = df[df["rating"] == 1]  # 평점이 1인 항목만 필터링
        df["item"] = df["item"] + n_user
        lines = df.groupby("user")["item"].apply(list)  # 사용자별로 아이템 리스트 생성

        for user, items in lines.items():
            user_item_dict[user] = items  # 사용자 ID를 키로 하고, 아이템 리스트를 값으로 저장

        return user_item_dict

    submission_rating_matrix = df_to_matrix(joined_rating_df)

    #########################################################################################

    # 3. Model
    print(f"--------------- INIT {args.model_name} ---------------")
    input_dims = [n_user, n_item, n_genre]
    model = getattr(model_module, args.model_name)(input_dims, **args.model_args[args.model_name]).to(args.device)

    print(f"--------------- PREDICT {args.model_name} ---------------")
    train_matrix = submission_rating_matrix
    trainer = getattr(trainer_module, args.model_name)(model, None, None, None, submission_loader, train_matrix, args)

    trainer.load(checkpoint_path)
    preds = trainer.submission(0)

    preds_df = pd.DataFrame(preds)
    preds_df.to_csv(os.path.join("output/", "preds.csv"), index=False)
    # preds = pd.read_csv("output/preds.csv")
    # preds = preds.values.tolist()

    reversed_users_dict = {v: k for k, v in users_dict.items()}
    reversed_items_dict = {(v + n_user): k for k, v in items_dict.items()}
    generate_submission_file(data_file, preds, reversed_items_dict)

    pred_df = pd.read_csv("/data/ephemeral/home/movie/davin/level2-recsys-movierecommendation-recsys-03-lv3/output/submission.csv")
    ans_df = pd.read_csv("/data/ephemeral/home/movie/davin/level2-recsys-movierecommendation-recsys-03-lv3/output/output.csv")
    common_items = pd.merge(pred_df, ans_df, on=['user', 'item'])
    num_common_items = common_items.shape[0]
    print("같은 user일 때 겹치는 item 수:", num_common_items)


if __name__ == "__main__":
    main()
