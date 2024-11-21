import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from dataset import ContextDataset
from preprocessing import filter_top_k_by_count, label_encoding, multi_hot_encoding, preprocess_title, tree2array, negative_sampling, pivot_count, merge_dataset, fill_na, df2mat

def load_dataset(args) -> pd.DataFrame:
    """
    데이터셋을 불러와 전처리 후 학습 데이터로 사용할 데이터프레임을 완성하는 함수

    Args:
        args (_type_): 

    Returns:
        pd.DataFrame: 전처리가 완료된 
    """
    # 학습 데이터 불러오기
    train_ratings = pd.read_csv(os.path.join(args.data_path, "train_ratings.csv"))

    # 아이템 side information 불러오기
    years = pd.read_csv(os.path.join(args.data_path, "years.tsv"), sep="\t")
    writers = pd.read_csv(os.path.join(args.data_path, "writers.tsv"), sep="\t")
    titles = pd.read_csv(os.path.join(args.data_path, "titles.tsv"), sep="\t")
    genres = pd.read_csv(os.path.join(args.data_path, "genres.tsv"), sep="\t")
    directors = pd.read_csv(os.path.join(args.data_path, "directors.tsv"), sep="\t")

    # 전처리: genre, director, writer의 상위 k개의 범주 레벨만 남기기
    # 이 때, 상위 k개는 빈도수 기준으로 내림차순으로 정해진다.
    _genres = filter_top_k_by_count(genres, sel_col="genre", pivot_col="item", top_k=4, ascending=args.preprocessing.ascending)
    _directors = filter_top_k_by_count(directors, sel_col="director", pivot_col="item", top_k=2)
    _writers = filter_top_k_by_count(writers, sel_col="writer", pivot_col="item", top_k=2)

    # 전처리: genre, director, writer 인코딩
    if args.preprocessing.encoding == "label":
        _genres = label_encoding(genres, label_col="genre", pivot_col="item")
    else:
        _genres = multi_hot_encoding(genres, label_col="genre", pivot_col="item")

    _directors = label_encoding(directors, label_col="director", pivot_col="item")
    _writers = label_encoding(writers, label_col="writer", pivot_col="item")

    # side information 데이터 item 기준으로 병합
    item_df = merge_dataset(titles, years, _genres, _directors, _writers)

    # 결측치 처리: side information 데이터를 병합하며서 생겨난 결측치 대체
    item_df = fill_na(item_df, col="director") # "unknown"으로 결측치 대체
    # item_df = fill_na(item_df, col="writer") # "unknown"으로 결측치 대체
    item_df = fill_na(item_df, col="year") # title의 괄호 안 연도를 추출해 결측치 대체

    # 전처리: 정규표현식 활용한 title 텍스트 전처리
    item_df = preprocess_title(item_df)

    # 전처리: 같은 영화인데 다른 item ID 값을 갖는 데이터 전처리
    train_ratings, item_df = replace_duplication(train_ratings, item_df)

    # 계층 구조 데이터프레임을 배열 구조 데이터프레임으로 변환
    if args.preprocessing.tree2array:
        item_df = tree2array(item_df, is_array=args.preprocessing.tree2array)
        
    # 파생변수 추가
    

    # train_ratings와 전처리가 끝난 item_df를 병합
    merged_train_df = pd.merge(train_ratings, item_df, on="item", how="left")

    # negative sampling

    return merged_train_df


# 전처리가 완료된 전체 학습 데이터 불러오기
merged_train_df = load_dataset(args)

users = list(set(merged_train_df.loc[:, "user"])).sort() # 유저 집합을 리스트로 생성
items = list(set(merged_train_df.loc[:, "item"])).sort() # 아이템 집합을 리스트로 생성
n_users = len(users)
n_items = len(items)

if (n_users - 1) != max(users):
    users_dict = {users[i]: i for i in range(n_users)}
    merged_train_df["user"]  = merged_train_df["user"].map(lambda x : users_dict[x])
    users = list(set(merged_train_df.loc[:,"user"]))

if (n_items - 1) != max(items):
    items_dict = {items[i]: i for i in range(n_items)}
    merged_train_df["item"]  = merged_train_df["item"].map(lambda x : items_dict[x])
    items =  list(set((merged_train_df.loc[:, "item"])))

merged_train_df = merged_train_df.sort_values(by=["user"])
merged_train_df.reset_index(drop=True, inplace=True)

# 데이터 분할
train_df, temp_df = train_test_split(merged_train_df, test_size=0.2, random_state=args.seed)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

# sparse matrix 생성    
valid_rating_matrix = df2mat(valid_df, merged_train_df)
test_rating_matrix = df2mat(test_df, merged_train_df)


def data_loader(data, batch_size, shuffle):
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

    dataset = ContextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

train_loader = data_loader(train_df, batch_size=1024, shuffle=True)
valid_loader = data_loader(valid_df, batch_size=512, shuffle=False)
test_loader = data_loader(test_df, batch_size=512, shuffle=False)