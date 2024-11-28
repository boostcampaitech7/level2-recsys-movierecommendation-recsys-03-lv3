# src/loader.py

import os

from argparse import Namespace
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import (
    filter_top_k_by_count,
    label_encoding,
    multi_hot_encoding,
    preprocess_title,
    negative_sampling,
    pivot_count,
    merge_dataset,
    fill_na,
    replace_id,
)


class Loader:
    def __init__(
        self,
        args,
    ):
        self.args = args

    def load_data():
        raise NotImplementedError


class DeepFM(Loader):
    def __init__(
        self,
        args,
    ):
        super(DeepFM, self).__init__(
            args,
        )

    def load_data(self):
        data_path = os.path.join(self.args.output_path, "merged_train_df.csv")
        if not os.path.isfile(data_path):
            merged_train_df = self.load_dataset(self.args)
            merged_train_df.to_csv(data_path, index=False)
        else:
            merged_train_df = pd.read_csv(data_path)

        merged_train_df, user_to_idx, item_to_idx = replace_id(merged_train_df)
        idx_to_user = {v: k for k, v in user_to_idx.items()}
        idx_to_item = {v + len(user_to_idx): k for k, v in item_to_idx.items()}

        merged_train_df.drop(columns=["title", "genre", "director", "time", "year", "num_reviews_item"], inplace=True)
        input_dims = merged_train_df.drop(columns=["review"]).nunique().tolist()
        self.args.model_args.DeepFM.input_dims = input_dims

        X_train, X_valid, y_train, y_valid = self.data_split(self.args, merged_train_df)
        X_all, y_all = merged_train_df.drop(columns=["review"]), merged_train_df["review"]

        seen_data = X_train.loc[y_train[y_train == 1].index]
        seen_data["item"] = seen_data["item"] + len(idx_to_user)
        seen_items = seen_data.groupby("user")["item"].apply(list).to_dict()

        train_loader = self.data_loader(["user", "item"], 1024, X_train, y_train, True)
        valid_loader = self.data_loader(["user", "item"], 512, X_valid, y_valid, False)
        test_loader = self.data_loader(["user", "item"], 1024, X_all, y_all, False)

        return train_loader, valid_loader, test_loader, seen_items, idx_to_user, idx_to_item, list(merged_train_df.columns)

    def load_dataset(self, args: Namespace) -> pd.DataFrame:
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
        item_df = fill_na(args, item_df, col="director")  # 계층 구조면 -1, 배열 구조면 [-1]로 결측치 대체
        # item_df = fill_na(item_df, col="writer")  # 계층 구조면 -1, 배열 구조면 [-1]로 결측치 대체
        item_df = fill_na(args, item_df, col="year")  # title의 괄호 안 연도를 추출해 결측치 대체

        # 전처리: 정규표현식 활용한 title 텍스트 전처리
        item_df = preprocess_title(item_df)

        # 전처리: 같은 영화인데 다른 item ID 값을 갖는 데이터 전처리
        # train_ratings, item_df = replace_duplication(train_ratings, item_df)

        # 계층 구조 데이터프레임을 배열 구조 데이터프레임으로 변환
        # if args.preprocessing.tree2array:
        #     item_df = tree2array(item_df, is_array=args.preprocessing.tree2array)

        # 전처리가 끝난 train_ratings와 item_df를 병합
        merged_train_df = pd.merge(train_ratings, item_df, on="item", how="left")

        # negative sampling
        if args.preprocessing.negative_sampling:
            merged_train_df = negative_sampling(merged_train_df, "user", "item", num_negative=50, na_list=merged_train_df.columns[3:])

        # 파생변수 추가: 아이템별 리뷰 수(num_reviews_item)
        merged_train_df = pivot_count(merged_train_df, pivot_col="item", col_name="num_reviews_item")

        # (user, item, time)이 중복되는 경우 제거
        # 같은 유저가 같은 아이템을 재평가(2번 이상 평가)한 사실을 시간이 다른 것으로 확인할 수 있었다.
        if args.preprocessing.is_array:
            merged_train_df = merged_train_df.drop_duplicates(["user", "item", "time"], ignore_index=True)

        return merged_train_df

    def data_split(self, args: Namespace, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # 사용자별로 데이터를 그룹화
        grouped = data.groupby("user")

        train_list = []
        valid_list = []

        # 각 사용자 그룹에서 훈련과 검증 세트를 나누기
        for _, group in grouped:
            if args.preprocessing.negative_sampling:
                X_train, X_valid, y_train, y_valid = train_test_split(
                    group.drop(columns="review"),
                    group["review"],
                    test_size=0.2,
                    random_state=args.seed,
                    shuffle=True
                )
                train_list.append(pd.concat([X_train, y_train], axis=1))
                valid_list.append(pd.concat([X_valid, y_valid], axis=1))
            else:
                X_train, X_valid = train_test_split(
                    group,
                    test_size=0.2,
                    random_state=args.seed,
                    shuffle=True
                )
                train_list.append(X_train)
                valid_list.append(X_valid)

        # 리스트를 데이터프레임으로 결합
        X_train = pd.concat(train_list, ignore_index=True)
        X_valid = pd.concat(valid_list, ignore_index=True)

        # y 값이 포함된 경우, y_train과 y_valid를 생성
        if args.preprocessing.negative_sampling:
            y_train = X_train.pop("review")
            y_valid = X_valid.pop("review")
            return X_train, X_valid, y_train, y_valid
        else:
            return X_train, X_valid

    def data_loader(
        self,
        cat_features: list[str],
        batch_size: int,
        X_data: pd.DataFrame = None,
        y_data: pd.DataFrame = None,
        shuffle: bool = True
    ) -> DataLoader:
        """
        최적화된 데이터 로더 생성 함수

        Args:
            cat_feautures (list): 범주형 변수 이름 리스트
            batch_size (int): 배치의 크기
            X_data (pd.DataFrame): X 데이터프레임. 디폴트는 None
            y_data (pd.DataFrame): y 데이터프레임. 디폴트는 None
            shuffle (bool): 데이터 셔플 유무. 디폴트는 True

        Returns:
            DataLoader: torch.utils의 DataLoader 객체
        """
        # NumPy 배열로 변환 (전체 데이터 한 번에 처리)
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
        if batch_size == 0:
            batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader


class EASE(Loader):
    def __init__(
        self,
        args,
    ):
        super(EASE, self).__init__(
            args,
        )

    def load_data(self) -> tuple[csr_matrix, csr_matrix, csr_matrix, dict[int, int], dict[int, int], int, int]:
        # 데이터 불러오기
        ratings = pd.read_csv(os.path.join(self.args.data_path, "train_ratings.csv"))

        # 데이터 분할
        train_data, test_data = self.train_test_split(ratings, split_ratio=0.8)

        # 원본/학습/테스트 데이터의 유저, 아이템 ID를 인덱스로 변환
        ratings, user_to_idx, item_to_idx, num_users, num_items = self.id2idx(ratings)
        train_data, _, _, _, _ = self.id2idx(train_data)
        test_data, _, _, _, _ = self.id2idx(test_data)
        self.args.model_args.MultiVAE.input_dim = num_items

        # 원본/학습/테스트 데이터의 희소 행렬 생성
        interaction_matrix = self._df2mat(ratings, num_users, num_items)
        train_data = self._df2mat(train_data, num_users, num_items)
        test_data = self._df2mat(test_data, num_users, num_items)

        # 이후 인덱스를 유저, 아이템 ID로 되돌리기 위한 딕셔너리 저장
        idx_to_user, idx_to_item = self.idx2id(user_to_idx, item_to_idx)

        return train_data, test_data, interaction_matrix, train_data, idx_to_user, idx_to_item, ["user", "item"]

    def train_test_split(self, data: pd.DataFrame, split_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        주어진 데이터를 유저별로 시간순으로 정렬 후, 1 - split_ratio만큼 최근 데이터를 테스트 데이터로 분할하는 함수

        Args:
            data (pd.DataFrame): 유저(user), 아이템(item), 평가 시간(time) 정보를 열로 포함하는 상호작용 데이터프레임
            split_ratio (float, optional): train_data로 만들 비율. 디폴트는 0.8

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 분할된 학습/테스트 데이터프레임
        """
        train_data = []
        test_data = []

        # 유저별 시간순으로 split_ratio만큼 train data로, 1 - split_ratio만큼 test data로 분할
        for _, group in data.groupby("user"):
            split_idx = int(len(group) * split_ratio)  # 제공된 데이터는 이미 시간순으로 정렬되어 있는 데이터
            train_data.append(group.iloc[:split_idx])
            test_data.append(group.iloc[split_idx:])
        train_data = pd.concat(train_data).reset_index(drop=True)
        test_data = pd.concat(test_data).reset_index(drop=True)
        return train_data, test_data

    def id2idx(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int], int, int]:
        # 유저와 아이템의 unique 인덱스 저장
        unique_users = data["user"].unique()
        unique_items = data["item"].unique()

        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

        # 유저와 아이템 ID를 unique 인덱스로 변환
        data["user"] = data["user"].map(user_to_idx)
        data["item"] = data["item"].map(item_to_idx)

        num_users = len(user_to_idx)
        num_items = len(item_to_idx)

        return data, user_to_idx, item_to_idx, num_users, num_items

    def idx2id(self, user_to_idx: dict[int, int], item_to_idx: dict[int, int]) -> tuple[dict[int, int], dict[int, int]]:
        # unique 인덱스를 유저, 아이템 ID로 변환
        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        return idx_to_user, idx_to_item

    def _df2mat(self, data: pd.DataFrame, num_users: int, num_items: int) -> csr_matrix:
        # 희소 행렬 생성
        rows, cols = data["user"].values, data["item"].values
        data = np.ones(len(data))
        interaction_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
        return interaction_matrix


class EASER(EASE):
    def __init__(
        self,
        args,
    ):
        super(EASER, self).__init__(
            args,
        )


class MultiVAE(EASE):
    def __init__(
        self,
        args,
    ):
        super(MultiVAE, self).__init__(
            args,
        )


class ContextDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.long()
        self.y = y.long()

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, index: int):
        if self.y is None:
            return self.X[index]
        else:
            return self.X[index], self.y[index]

