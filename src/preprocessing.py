import os
import re
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from typing import Tuple
from scipy.sparse import csr_matrix


# 피벗별로 상위 k개의 레벨만 남기기
def filter_top_k_by_count(
        df: pd.DataFrame,
        sel_col: str,
        pivot_col: str,
        top_k: int,
        ascending: bool = False
    ) -> pd.DataFrame:
    """아이템(유저)별 범주를 빈도 순으로 k개만 추출합니다.

    Args:
        df (pd.DataFrame): 원본 데이터프레임
        sel_col (str): 범주형 데이터 열 이름
        pivot_col (str): 기준으로 할 열 이름
        top_k (int): 몇 개를 추출할지 결정하는 정수
        ascending (bool, optional): 추출 기준을 오름차순으로 할 지 여부. False가 기본값임.

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
    """데이터프레임에 라벨 인코딩을 적용합니다.

    Args:
        df (pd.DataFrame): 원본 데이터프레임
        label_col (str): 인코딩을 할 열의 이름
        pivot_col (str): 배열 형태로 나타낼 때의 기준 열의 이름. None으로 입력하면 계층적 표현으로 반환.

    Returns:
        pd.DataFrame: 인코딩이 적용된 데이터프레임
    """
    # 범주형 자료를 수치형으로 변환
    array, _ = pd.factorize(df[label_col])
    
    # 변환된 값으로 새로운 데이터프레임 생성
    tmp_df = df.copy()
    tmp_df[label_col] = array

    if pivot_col != None:
        # 리스트 형태로 변환 후 데이터프레임 반환
        grouped_df = tmp_df.groupby(pivot_col)[label_col].apply(list)
        result_df = pd.merge(tmp_df["item"], grouped_df, on="item", how="left")
        return result_df
    
    return tmp_df

# 함수 정의: 멀티-핫-인코딩 하기
def multi_hot_encoding(df: pd.DataFrame,
                       label_col: str,
                       pivot_col: str
                       ) -> pd.DataFrame:
    """
    범주형 데이터에서 여러 개의 선택 가능한 값을 이진 벡터(binary vector)로 변환합니다.

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
    """정규 표현식을 이용해 title 변수의 텍스트를 전처리합니다.

    Args:
        df (pd.DataFrame): 원본 데이터프레임
        col (str): 전처리할 열 이름. title이 기본값이며 이외에 사용하지 않음.

    Returns:
        pd.DataFrame: 전처리가 완료된 데이터프레임
    """
    # 1. 따옴표(”, ‘) 제거
    df[col] = re.sub(r'^[\'"](.*)[\'"]$', r'\1', df[col])
    
    # 2. 영문 제목만 추출
    df[col] = re.match(r'^[^(]+', df[col]).group().strip() if re.match(r'^[^(]+', df[col]) else df[col]
    
    # 3. "~, The", "~, A", "~, An" 형태를 "The ~", "A ~", "An ~"으로 변경
    df[col] = re.sub(r'^(.*),\s(The|A|An)$', r'\2 \1', df[col])
    
    # 4. 특수문자 제거
    df[col] = re.sub(r'[^a-zA-Z0-9\s]', '', df[col])
    
    # 5. 소문자로 변환
    df[col] = df[col].lower()
    
    return df

def fill_na(
        df: pd.DataFrame,
        col: str,
    ) -> pd.DataFrame:
    match col:
        case "year":
            df[col] = df[col].fillna(
                df["title"].str.extract(r"\((\d{4})\)", expand=False)  # 괄호 안 네 자리 숫자를 추출하는 정규표현식
            ).astype("int64")

        case _:
            df[col] = df[col].fillna("unknown")
    
    return df

def replace_duplication(
        train_ratings: pd.DataFrame,
        item_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """중복되는 아이템이 아이템 번호가 같은 것을 제거하고, rating 데이터프레임에서는 하나의 아이템 번호로 통합합니다.
    현재 우주전쟁(War of the Worlds, 2005) 영화가 2개의 item ID를 갖고 있습니다. 따라서 같은 영화인데 다른 item 값을 갖는 데이터 중에서 결측치가 있는 item 제거합니다.

    Args:
        train_ratings (pd.DataFrame): train_ratings 데이터프레임
        item_df (pd.DataFrame): item_df 데이터프레임

    Returns:
        Tuple
        - pd.DataFrame: 전처리가 완료된 train_ratings 데이터프레임
        - pd.DataFrame: 전처리가 완료된 item_df 데이터프레임 
    """

    item_df = item_df[item_df["item"] != 64997]

    # train_ratings에서 item 값을 변경하려는 인덱스 추출
    idx = train_ratings[(train_ratings["item"] == 64997)].index

    # train_ratings에 원하는 item 값으로 변경
    train_ratings.loc[idx, "item"] = 34048

    return train_ratings, item_df

# 계층 구조로 이루어진 데이터프레임을 배열 구조로 이루어진 데이터프레임으로 변경하는 함수
def tree2array(df: pd.DataFrame, is_array: bool) -> pd.DataFrame:
    """계층적 구조로 이루어진 데이터프레임을 범주 리스트 형식으로 변환합니다.

    Args:
        df (pd.DataFrame): 원본 데이터프레임
        is_array (bool): 장르 변수를 배열 형태로 나타낼지 여부

    Returns:
        pd.DataFrame: 변환된 데이터프레임
    """
    rule = {
        "director": lambda x: list(x.unique()),
        "writer": lambda x: list(x.unique())
    }
    if is_array:
        rule["genre"] = lambda x: list(x.unique())
    df_tolist = df.groupby(["item", "title", "year"]).agg(rule).reset_index()

    return df_tolist

# 함수 정의: num_negative만큼 negative_sampling하기
def negative_sampling(
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        num_negative: int
    ) -> pd.DataFrame:
    """
    주어진 데이터프레임의 각 사용자에 대해 부정 샘플을 생성하고, 긍정 샘플과 결합하여 최종 데이터프레임을 반환합니다.

    Args:
        df (pd.DataFrame): user_col과 item_col을 column으로 갖는 데이터프레임
        user_col (str):  데이터프레임에서 사용자 ID를 나타내는 변수명
        item_col (str): 데이터프레임에서 아이템 ID를 나타내는 변수명
        num_negative (int): negative sample의 수

    Returns:
        pd.DataFrame: 기존 데이터프레임에 부정 샘플까지 결합한 데이터프레임 반환
    """

    df['review'] = 1
    user_group_dfs = list(df.groupby(user_col)[item_col])
    first_row = True
    user_neg_dfs = pd.DataFrame()
    items = set(df.loc[:, item_col])

    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)
        i_user_neg_df = pd.DataFrame({user_col: [u]*num_negative, item_col: i_user_neg_item, 'review': [0]*num_negative})
        
        if first_row == True:
            user_neg_dfs = i_user_neg_df
            first_row = False
        
        else:
            user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)

    raw_rating_df = pd.concat([df, user_neg_dfs], axis = 0, sort=False) 
    
    return raw_rating_df
 
# pivot_col 기준으로 카운팅하기
def pivot_count(df: pd.DataFrame,
                pivot_col: str,
                col_name: str,
                ) -> pd.DataFrame:
    """
    주어진 데이터프레임에서 특정 열의 값에 대한 카운트를 계산하고, 그 결과를 새로운 열로 추가하여 최종 데이터프레임을 반환합니다.
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        pivot_col (str): 데이터프레임에서 피벗할 변수명
        col_name (str): pivot_col에서 계산된 카운트 값을 포함할 새로운 변수명

    Returns:
        pd.DataFrame: 최종 데이터프레임을 반환
    """

    if 'review' in df.columns:
        positive_df =  df[df["review"]==1]
        pivot_count_df = positive_df[pivot_col].value_counts()
    
    else:
        pivot_count_df = df[pivot_col].value_counts()

    df[col_name] =  df[pivot_col].map(pivot_count_df)
    
    return df

def merge_dataset(
        titles: pd.DataFrame, 
        years: pd.DataFrame, 
        genres: pd.DataFrame, 
        directors: pd.DataFrame, 
        writers: pd.DataFrame
    ) -> pd.DataFrame:
    """
    side information을 하나의 item 데이터프레임으로 병합합니다.

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
    # item_df = pd.merge(item_df, writers, on="item", how="left")
    return item_df

# sparse matrix 생성
def df2mat(df: pd.DataFrame, merged_df: pd.DataFrame) -> csr_matrix:
    """데이터프레임의 user-item matrix를 생성합니다.

    Args:
        df (pd.DataFrame): 훈련/검증 데이터(리뷰)
        merged_df (pd.DataFrame): 훈련 데이터와 검증 데이터로 분할하기 전 데이터프레임

    Returns:
        scipy.sparse.csr_matrix: 희소 행렬 형태의 user-item matrix
    """
    # 유저, 아이템을 zero-based index로 매핑: sparse matrix shape 조절
    users = list(set(merged_df.loc[:, "user"])).sort() # 유저 집합을 리스트로 생성
    items = list(set(merged_df.loc[:, "item"])).sort() # 아이템 집합을 리스트로 생성
    n_users = len(users)
    n_items = len(items)

    if (n_users - 1) != max(users):
        users_dict = {users[i]: i for i in range(n_users)}
        merged_df["user"]  = merged_df["user"].map(lambda x : users_dict[x])
        users = list(set(merged_df.loc[:,"user"]))

    if (n_items - 1) != max(items):
        items_dict = {items[i]: i for i in range(n_items)}
        merged_df["item"]  = merged_df["item"].map(lambda x : items_dict[x])
        items =  list(set((merged_df.loc[:, "item"])))

    merged_df = merged_df.sort_values(by=["user"])
    merged_df.reset_index(drop=True, inplace=True)


    if "review" in df.columns: df = df[df["review"]]

    user_seq = []
    lines = df.groupby("user")["item"].apply(list)
    for line in lines:
        user_seq.append(line.values)

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
    rating_matrix = csr_matrix(
        (data, (row, col)), shape=(merged_df["user"].nunique(), merged_df["item"].nunique())
    )

    return rating_matrix