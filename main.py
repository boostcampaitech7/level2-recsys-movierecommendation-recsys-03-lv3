import os

import argparse
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import wandb
import json
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from src.utils import set_seed, check_path, EarlyStopping
import src.model as model_module
import src.trainer as trainer_module


def main():

    # 1. Basic Environment Setup
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "--m", default="DeepFM", type=str,
                        help="사용할 모델을 설정할 수 있습니다. (기본값 DeepFM)")
    parser.add_argument("--epochs", "--e", default=2, type=int,
                        help="모델 훈련을 반복할 epochs수를 지정할 수 있습니다. (기본값 2)")
    parser.add_argument("--optuna", "--o", default=False, type=bool,
                        help="Optuna 사용 여부를 설정할 수 있습니다. (기본값 False)")
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
        # settings=wandb.Settings(code_dir="./src")  
    )

    # Artifacts에 폴더 내 파일 모두 업로드
    wandb.run.log_code("./src")

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
    print(raw_genre_df)

    genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df["genre"]))}
    raw_genre_df["genre"]  = raw_genre_df["genre"].map(lambda x : genre_dict[x]) #genre id로 변경

    joined_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on="item", right_on="item", how="inner")
    users = list(set(joined_rating_df.loc[:,"user"]))
    users.sort()
    items =  list(set((joined_rating_df.loc[:, "item"])))
    items.sort()
    genres =  list(set((joined_rating_df.loc[:, "genre"])))
    genres.sort()

    if len(users)-1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        joined_rating_df["user"]  = joined_rating_df["user"].map(lambda x : users_dict[x])
        users = list(set(joined_rating_df.loc[:,"user"]))

    if len(items)-1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        joined_rating_df["item"]  = joined_rating_df["item"].map(lambda x : items_dict[x])
        items =  list(set((joined_rating_df.loc[:, "item"])))

    joined_rating_df = joined_rating_df.sort_values(by=["user"])
    joined_rating_df.reset_index(drop=True, inplace=True)

    # ################### split

    # train_ratio = 0.8
    # valid_ratio = 0.1

    # total_size = len(joined_rating_df)
    # train_size = int(train_ratio * total_size)
    # valid_size = int(valid_ratio * total_size)
    # test_size = total_size - train_size - valid_size

    # train_df, temp_df = random_split(joined_rating_df, [train_size, valid_size + test_size])
    # valid_df, test_df = random_split(temp_df, [valid_size, test_size])
    train_df, temp_df = train_test_split(joined_rating_df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    ################### matrix

    def df_to_matrix(df):
        user_seq = []
        lines = df.groupby("user")["item"].apply(list)
        for line in lines:
            items = line
            user_seq.append(items)

        row = []
        col = []
        data = []
        for user_id, item_list in enumerate(user_seq):
            for item in item_list: 
                row.append(user_id)
                col.append(item)
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix((data, (row, col)), shape=(joined_rating_df["user"].nunique(), joined_rating_df["item"].nunique()))
        return rating_matrix
    
    valid_rating_matrix = df_to_matrix(valid_df)
    test_rating_matrix = df_to_matrix(test_df)

    # # 3. Negative instance 생성
    # print("Create Nagetive instances")
    # def negative_sampling(raw_rating_df):
    #     num_negative = 50
    #     user_group_dfs = list(raw_rating_df.groupby("user")["item"])
    #     first_row = True
    #     user_neg_dfs = pd.DataFrame()

    #     items =  set((raw_rating_df.loc[:, "item"]))
    #     for u, u_items in tqdm(user_group_dfs):
    #         u_items = set(u_items)
    #         i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)

    #         i_user_neg_df = pd.DataFrame({"user": [u]*num_negative, "item": i_user_neg_item, "rating": [0]*num_negative})
    #         if first_row == True:
    #             user_neg_dfs = i_user_neg_df
    #             first_row = False
    #         else:
    #             user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)

    #     raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis = 0, sort=False)
    #     return raw_rating_df

    # train_df = negative_sampling(train_df)
    # valid_df = negative_sampling(valid_df)
    # test_df = negative_sampling(test_df)
    

    # train_df.to_csv(os.path.join("output/", "train_df.csv"), index=False)
    # valid_df.to_csv(os.path.join("output/", "valid_df.csv"), index=False)
    # test_df.to_csv(os.path.join("output/", "test_df.csv"), index=False)
    # test_df.to_csv(os.path.join("output/", "joined_rating_df.csv"), index=False)
    train_df = pd.read_csv("output/train_df.csv")
    valid_df = pd.read_csv("output/valid_df.csv")
    test_df = pd.read_csv("output/test_df.csv")
    # joined_rating_df = pd.read_csv("output/joined_rating_df.csv")

    ################### dataloader

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
    
    train_loader = df_to_dataloader(train_df, 1024, True)
    valid_loader = df_to_dataloader(valid_df, 512, False)
    test_loader = df_to_dataloader(test_df, 512, False)

    #########################################################################################

    # 3. Model
    print(f"--------------- INIT {args.model_name} ---------------")
    input_dims = [n_user, n_item, n_genre]
    model = getattr(model_module, args.model_name)(input_dims, **args.model_args[args.model_name]).to(args.device)

    # 4. Train
    print(f"--------------- {args.model_name} TRAINING ---------------")
    train_matrix = valid_rating_matrix
    trainer = getattr(trainer_module, args.model_name)(model, train_loader, valid_loader, test_loader, None, train_matrix, df, args)
    
    checkpoint = args_str + ".pt"
    checkpoint_path = os.path.join(args.output_path, checkpoint)
    early_stopping = EarlyStopping(checkpoint_path, patience=10, verbose=True)
    for epoch in tqdm(range(args.epochs)):
        trainer.train(epoch)
        scores = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 5. Test
    print(f"--------------- {args.model_name} TEST ---------------")
    trainer.train_matrix = test_rating_matrix
    trainer.train_df = train_df
    trainer.model.load_state_dict(torch.load(checkpoint_path))
    _ = trainer.test(0)

    wandb.log({
        "params": json.dumps(OmegaConf.to_container(args.model_args[args.model_name]))
    })
    wandb.finish()


if __name__ == "__main__":
    main()
