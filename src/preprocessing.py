import re
from argparse import Namespace
from collections import Counter

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


# 피벗별로 상위 k개의 레벨만 남기기
def filter_top_k_by_count(
        df: pd.DataFrame,
        sel_col: str,
        pivot_col: str,
        top_k: int,
        ascending: bool = False
    ) -> pd.DataFrame:
    """
    아이템(유저)별 범주를 빈도 순으로 k개만 추출하는 함수

    Args:
        df (pd.DataFrame): 원본 데이터프레임
        sel_col (str): 범주형 데이터 column 이름
        pivot_col (str): 기준으로 할 column 이름
        top_k (int): 몇 개를 추출할지 결정하는 정수
        ascending (bool, optional): 추출 기준을 오름차순으로 할 지 여부. 기본값은 False.

    Returns:
        pd.DataFrame: 전처리가 완료된 데이터프레임
    """
    # 1. 레벨별 전체 등장 빈도 계산
    col_count = df[sel_col].value_counts().reset_index()
    col_count.columns = [sel_col, "count"]

    # 2. 원본 데이터프레임에 레벨 count 추가
    df = df.merge(col_count, on=sel_col)

    # 3. 피벗에서 상위 k개의 레벨 남기기
    filtered_df = df.groupby(pivot_col).apply(
        (lambda x: x.nsmallest(top_k, "count")) if ascending else (lambda x: x.nlargest(top_k, "count"))
    ).reset_index(drop=True)

    # 4. count 열 제거 후 결과 반환
    result_df = filtered_df.drop(columns=["count"])

    return result_df


def label_encoding(
        df: pd.DataFrame,
        label_col: str,
        pivot_col: str = None,
    ) -> pd.DataFrame:
    """
    범주형 데이터에 라벨 인코딩을 적용하는 함수

    Args:
        df (pd.DataFrame): 원본 데이터프레임
        label_col (str): 인코딩을 할 column의 이름
        pivot_col (str): 배열 형태로 나타낼 때의 기준 column의 이름. None으로 입력하면 계층적 표현으로 반환.

    Returns:
        pd.DataFrame: 인코딩이 적용된 데이터프레임
    """
    # 범주형 자료를 수치형으로 변환
    array, _ = pd.factorize(df[label_col])

    # 변환된 값으로 새로운 데이터프레임 생성
    tmp_df = df.copy()
    tmp_df[label_col] = array

    if pivot_col is None:
        return tmp_df
    else:
        # 리스트 형태로 변환 후 데이터프레임 반환
        grouped_df = tmp_df.groupby(pivot_col)[label_col].apply(list)
        result_df = pd.merge(tmp_df["item"], grouped_df, on="item", how="left")
        
        return result_df



def multi_hot_encoding(
        df: pd.DataFrame,
        label_col: str,
        pivot_col: str
    ) -> pd.DataFrame:
    """
    범주형 데이터에 멀티-핫 인코딩을 적용하는 함수

    Args:
        df (pd.DataFrame): pivot_col과 label_col을 column으로 갖는 데이터프레임
        label_col (str): 데이터프레임에서 멀티 핫 인코딩을 적용할 범주형 변수명
        pivot_col (str): 데이터프레임에서 그룹화할 기준이 되는 변수명

    Returns:
        pd.DataFrame: 멀티-핫-인코딩이 완료된 데이터프레임 반환.
    """

    # 1. pivot_col별 label_col을 리스트로 묶기
    grouped_df = df.groupby(pivot_col)[label_col].apply(lambda x: list(x)).reset_index()

    # 2. MultiLabelBinarizer를 사용하여 멀티 핫 인코딩 수행
    mlb = MultiLabelBinarizer()
    multi_hot_encoded = mlb.fit_transform(grouped_df[label_col])

    # 3. 결과를 데이터프레임으로 변환
    multi_hot_df = pd.DataFrame(multi_hot_encoded, columns=mlb.classes_)

    # 4. 원본 데이터프레임과 결합
    result_df = pd.concat([grouped_df[pivot_col], multi_hot_df], axis=1)

    return result_df


# 정규표현식을 활용한 title 텍스트 전처리 함수
def preprocess_title(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """
    정규 표현식을 이용해 title 변수의 텍스트를 전처리하는 함수

    Args:
        df (pd.DataFrame): 원본 데이터프레임
        col (str): 전처리할 열 이름. title이 기본값이며 이외에 사용하지 않음.

    Returns:
        pd.DataFrame: 전처리가 완료된 데이터프레임
    """
    # 1. 따옴표(”, ‘) 제거
    df[col] = df[col].apply(lambda x: re.sub(r'^[\'"](.*)[\'"]$', r'\1', x))
    
    # 2. 영문 제목만 추출
    df[col] = df[col].apply(lambda x: re.match(r'^[^(]+', x).group().strip() if re.match(r'^[^(]+', x) else x)
    
    # 3. "~, The", "~, A", "~, An" 형태를 "The ~", "A ~", "An ~"으로 변경
    df[col] = df[col].apply(lambda x: re.sub(r'^(.*),\s(The|A|An)$', r'\2 \1', x))
    
    # 4. 특수문자 제거
    df[col] = df[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    
    # 5. 소문자로 변환
    df[col] = df[col].apply(lambda x: x.lower())

    return df


def fill_na(
        args: Namespace,
        df: pd.DataFrame,
        col: str,
    ) -> pd.DataFrame:
    """
    side information을 병합하면서 생겨난 결측치를 처리하는 함수

    Args:
        args (Namespace): parser.parse_args()에서 반환되는 Namespace 객체. is_array에 따라 결측치 처리 방법을 달리 한다.
        df (pd.DataFrame): 결측치를 갖는 데이터프레임
        col (str): 결측치를 처리하고자 하는 열 이름

    Returns:
        pd.DataFrame: 결측치를 처리한 데이터프레임
    """
    match col:
        case "year":
            df[col] = df[col].fillna(
                df["title"].str.extract(r"\((\d{4})\)", expand=False)  # 괄호 안 네 자리 숫자를 추출하는 정규표현식
            ).astype("int64")

        case _:
            df[col] = df[col].fillna(-1)
            if args.preprocessing.is_array:
                df[col] = df[col].apply(lambda x: x if type(x) is list else [x])

    return df


def negative_sampling(
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        num_negative: int,
        na_list: list[str, str, str, str, str] = ["title", "year", "genre", "director", "writer"]
    ) -> pd.DataFrame:
    """
    주어진 데이터프레임의 각 사용자에 대해 negative sample을 생성하고, positive sample과 결합하여 최종 데이터프레임을 반환하는 함수

    Args:
        df (pd.DataFrame): user_col과 item_col을 column으로 갖는 데이터프레임
        user_col (str):  데이터프레임에서 사용자 ID를 나타내는 변수명
        item_col (str): 데이터프레임에서 아이템 ID를 나타내는 변수명
        num_negative (int): negative sample의 수
        na_list (list[str, str, str, str]): negative sampling 이후 결측치를 처리할 column의 이름

    Returns:
        pd.DataFrame: 기존 데이터프레임에 negative sample까지 결합한 데이터프레임
    """
    # 아이템 전체 집합 및 사용자별 아이템 목록 미리 생성
    items = set(df[item_col].unique())
    user_items_dict = df.groupby(user_col)[item_col].apply(set).to_dict()

    # na_list 컬럼 데이터를 아이템별로 미리 매핑
    item_na_map = {col: df.groupby(item_col)[col].first().to_dict() for col in na_list}

    neg_samples = []

    for user, u_items in tqdm(user_items_dict.items()):
        # 사용자가 이미 본 아이템 제외하고 negative sample 선택
        negative_items = np.random.choice(list(items - u_items), num_negative, replace=False)

        # negative sample 데이터 생성
        for item in negative_items:
            neg_sample = {user_col: user, item_col: item, "review": 0}
            for na_col in na_list:
                neg_sample[na_col] = item_na_map[na_col].get(item, None)
            neg_samples.append(neg_sample)

    # negative sample과 기존 데이터 결합
    neg_samples_df = pd.DataFrame(neg_samples)
    raw_rating_df = pd.concat([df, neg_samples_df], axis=0, ignore_index=True)
    raw_rating_df["review"] = raw_rating_df["review"].fillna(1)
    raw_rating_df["review"] = raw_rating_df["review"].astype("int64")
    raw_rating_df["time"] = raw_rating_df["time"].fillna(0)
    
    if raw_rating_df.isna().sum().sum():
        raise ValueError("처리되지 않은 결측치가 있습니다.")

    return raw_rating_df


# pivot_col 기준으로 카운팅하기
def pivot_count(
        df: pd.DataFrame,
        pivot_col: str,
        col_name: str,
    ) -> pd.DataFrame:
    """
    주어진 데이터프레임에서 pivot_col의 값을 카운팅한 값을 데이터프레임에 col_name으로 추가하는 함수

    Args:
        df (pd.DataFrame): 데이터프레임
        pivot_col (str): 빈도 계산할 기준이 되는 column
        col_name (str): 데이터프레임에 새로 추가될 column 이름

    Returns:
        pd.DataFrame: col_name이 추가된 데이터프레임
    """
    if "review" in df.columns:
        positive_df = df[df["review"] == 1]
        pivot_count_df = positive_df[pivot_col].value_counts()
    else:
        pivot_count_df = df[pivot_col].value_counts()

    df[col_name] = df[pivot_col].map(pivot_count_df)

    return df


def merge_dataset(
        titles: pd.DataFrame,
        years: pd.DataFrame,
        genres: pd.DataFrame,
        directors: pd.DataFrame,
        writers: pd.DataFrame
    ) -> pd.DataFrame:
    """
    side information을 하나의 item 데이터프레임으로 병합하는 함수

    Args:
        titles (pd.DataFrame): 아이템 ID(item)와 제목(title) 정보를 담고있는 데이터프레임
        years (pd.DataFrame): 아이템 ID(item)와 개봉연도(year) 정보를 담고있는 데이터프레임
        genres (pd.DataFrame): 아이템 ID(item)와 장르(genre) 정보를 담고있는 데이터프레임
        directors (pd.DataFrame): 아이템 ID(item)와 감독(director) 정보를 담고있는 데이터프레임
        writers (pd.DataFrame): 아이템 ID(item)와 작가(writer) 정보를 담고있는 데이터프레임

    Returns:
        pd.DataFrame: side information을 모두 합친 데이터프레임
    """
    item_df = pd.merge(titles, years, on="item", how="left")
    item_df = pd.merge(item_df, genres, on="item", how="left")
    item_df = pd.merge(item_df, directors, on="item", how="left")
    item_df = pd.merge(item_df, writers, on="item", how="left")

    return item_df


def replace_id(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """
    유저와 아이템 ID를 0부터 시작하는 nunique까지의 숫자로 매핑하는 함수

    Args:
        merged_df (pd.DataFrame): 유저와 아이템 ID(user, item)를 column으로 갖는 데이터프레임

    Returns:
        tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
        - pd.DataFrame: 유저와 아이템 ID가 변환된 데이터프레임
        - dict[int, int]: 기존 유저 ID를 key로, 매핑하려는 값을 value로 갖는 딕셔너리
        - dict[int, int]: 기존 아이템 ID를 key로, 매핑하려는 값을 value로 갖는 딕셔너리
    """
    # 유저, 아이템을 zero-based index로 매핑
    users = merged_df["user"].unique()  # 유저 집합을 리스트로 생성
    items = merged_df["item"].unique()  # 아이템 집합을 리스트로 생성
    n_users = len(users)
    n_items = len(items)

    if (n_users - 1) != max(users):
        users_dict = {users[i]: i for i in range(n_users)}
        merged_df["user"] = merged_df["user"].map(lambda x: users_dict[x])
        users = list(set(merged_df.loc[:, "user"]))

    if (n_items - 1) != max(items):
        items_dict = {items[i]: i for i in range(n_items)}
        merged_df["item"] = merged_df["item"].map(lambda x: items_dict[x])
        items = list(set((merged_df.loc[:, "item"])))

    merged_df = merged_df.sort_values(by=["user"])
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df, users_dict, items_dict
