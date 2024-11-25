import os

import numpy as np
import pandas as pd
import torch

from argparse import Namespace
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.dataset import ContextDataset
from src.preprocessing import (
    filter_top_k_by_count, 
    label_encoding, 
    multi_hot_encoding, 
    preprocess_title, 
    tree2array, 
    negative_sampling, 
    pivot_count, 
    merge_dataset, 
    fill_na, 
    replace_duplication
)

def load_dataset(args: Namespace) -> pd.DataFrame:
    """
    데이터셋을 불러와 전처리 후 학습 데이터로 사용할 데이터프레임을 완성하는 함수

    Args:
        args (Namespace): parser.parse_args()에서 반환되는 Namespace 객체

    Returns:
        pd.DataFrame: 전처리 완료 후 병합된 데이터프레임
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
    if args.preprocessing.is_array:
        if args.preprocessing.encoding == "label":
            _genres = label_encoding(genres, label_col="genre", pivot_col="item")
        else:
            _genres = multi_hot_encoding(genres, label_col="genre", pivot_col="item")
        _directors = label_encoding(directors, label_col="director", pivot_col="item")
        _writers = label_encoding(writers, label_col="writer", pivot_col="item")
    else:
        _genres = label_encoding(genres, label_col="genre")
        _directors = label_encoding(directors, label_col="director")
        _writers = label_encoding(writers, label_col="writer")
    
    # side information 데이터 item 기준으로 병합
    item_df = merge_dataset(titles, years, _genres, _directors, _writers)

    # 결측치 처리: side information 데이터를 병합하며서 생겨난 결측치 대체
    item_df = fill_na(args, item_df, col="director") # 계층 구조면 -1, 배열 구조면 [-1]로 결측치 대체
    # item_df = fill_na(item_df, col="writer") # 계층 구조면 -1, 배열 구조면 [-1]로 결측치 대체
    item_df = fill_na(args, item_df, col="year") # title의 괄호 안 연도를 추출해 결측치 대체

    # 전처리: 정규표현식 활용한 title 텍스트 전처리
    item_df = preprocess_title(item_df)

    # 전처리: 같은 영화인데 다른 item ID 값을 갖는 데이터 전처리
    train_ratings, item_df = replace_duplication(train_ratings, item_df)

    # 계층 구조 데이터프레임을 배열 구조 데이터프레임으로 변환
    # if args.preprocessing.tree2array:
    #     item_df = tree2array(item_df, is_array=args.preprocessing.tree2array)

    # 전처리가 끝난 train_ratings와 item_df를 병합
    merged_train_df = pd.merge(train_ratings, item_df, on="item", how="left")

    # negative sampling
    if args.preprocessing.negative_sampling:
        merged_train_df = negative_sampling(merged_train_df, "user", "item", num_negative=5, na_list=merged_train_df.columns[3:])
        
    # 파생변수 추가: 아이템별 리뷰 수(num_reviews_item)
    merged_train_df = pivot_count(merged_train_df, pivot_col="item", col_name="num_reviews_item")

    # (user, item, time)이 중복되는 경우 제거
    # 같은 유저가 같은 아이템을 재평가(2번 이상 평가)한 사실을 시간이 다른 것으로 확인할 수 있었다.
    if args.preprocessing.is_array:
        merged_train_df = merged_train_df.drop_duplicates(["user", "item", "time"], ignore_index=True)

    return merged_train_df


def data_split(args: Namespace, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.preprocessing.negative_sampling:
        X_train, X_valid, y_train, y_valid = train_test_split(
            data.drop(columns="review"),
            data["review"],
            test_size=0.2,
            random_state=args.seed,
            shuffle=True
        )
        return X_train, X_valid, y_train, y_valid    
    else:
        X_train, X_valid = train_test_split(
            data,
            test_size=0.2,
            random_state=args.seed,
            shuffle=True
        )
        return X_train, X_valid


def data_loader(
    cat_features: list[str],
    batch_size: int, 
    X_data: pd.DataFrame = None,
    y_data: pd.DataFrame = None,
    shuffle: bool = True
) -> DataLoader:
    """
    데이터 로더를 생성하는 함수

    Args:
        cat_feautures (list): 범주형 변수 이름 리스트
        batch_size (int): 배치의 크기
        X_data (pd.DataFrame): X 데이터프레임. 디폴트는 None
        y_data (pd.DataFrame): y 데이터프레임. 디폴트는 None
        shuffle (bool): 데이터 셔플 유무. 디폴트는 True

    Returns:
        DataLoader: torch.utils의 DataLoader 객체
    """
    # "title" 열 제거 후 NumPy 배열로 변환 (전체 데이터 한 번에 처리)
    X_data = X_data.drop("title", axis=1)
    cat_data = X_data.loc[:, cat_features]
    cat_data_np = cat_data.to_numpy()

    # 고유값 오프셋 계산
    unique_counts = [0] + np.cumsum([len(np.unique(cat_data_np[:, i])) for i in range(cat_data_np.shape[1])]).tolist()

    # 데이터 오프셋 추가 (벡터화 처리)
    for i in range(cat_data_np.shape[1]):
        cat_data_np[:, i] += unique_counts[i]

    # X와 y 텐서 변환
    X = torch.cat([
        torch.tensor(cat_data_np, dtype=torch.float32),
        torch.tensor(X_data.drop(cat_features, axis=1).values, dtype=torch.float32)
    ], axis=1)
    if y_data is not None:
        y = torch.tensor(y_data.values, dtype=torch.float32)
    else:
        y = None

    dataset = ContextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader