import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


# 피벗별로 상위 k개의 레벨만 남기기
def filter_top_k_by_count(df, sel_col, pivot_col, top_k, ascending=False):
    # 1. 레벨별 전체 등장 빈도 계산
    col_count = df[sel_col].value_counts().reset_index()
    col_count.columns = [sel_col, "count"]
    
    # 2. 원본 데이터프레임에 레벨 count 추가
    df = df.merge(col_count, on=sel_col)
    
    # 3. 각 피벗별로 상위 N개의 레벨 남기기
    filtered_df = df.groupby(pivot_col).apply(
        (lambda x: x.nsmallest(top_k, "count")) if ascending else (lambda x: x.nlargest(top_k, "count"))
    ).reset_index(drop=True)
    
    # 4. count 열 제거 후 결과 반환
    result_df = filtered_df.drop(columns=["count"])
    
    return result_df

def label_encoding(df, label_col, pivot_col, to_list=True):
    # 범주형 자료를 수치형으로 변환
    array, _ = pd.factorize(df[label_col])
    
    # 변환된 값으로 새로운 데이터프레임 생성
    # tmp_df = df.assign(**{label_col: array})
    tmp_df = df.copy()
    tmp_df[label_col] = array

    if to_list:
        # 리스트 형태로 변환 후 데이터프레임 반환
        grouped_df = tmp_df.groupby(pivot_col)[label_col].apply(list)
        result_df = pd.merge(tmp_df["item"], grouped_df, on="item", how="left")
        return result_df
    else:
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
def preprocess_title(title):
    # 1. 따옴표(”, ‘) 제거
    title = re.sub(r'^[\'"](.*)[\'"]$', r'\1', title)
    
    # 2. 영문 제목만 추출
    title = re.match(r'^[^(]+', title).group().strip() if re.match(r'^[^(]+', title) else title
    
    # 3. "~, The", "~, A", "~, An" 형태를 "The ~", "A ~", "An ~"으로 변경
    title = re.sub(r'^(.*),\s(The|A|An)$', r'\2 \1', title)
    
    # 4. 특수문자 제거
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    
    # 5. 소문자로 변환
    title = title.lower()
    
    return title

# 계층 구조로 이루어진 데이터프레임을 배열 구조로 이루어진 데이터프레임으로 변경하는 함수
def tree2array(df):
    df_tolist = df.groupby(["item", "title", "year"]).agg({
        "genre": lambda x: list(x.unique()),
        "director": lambda x: list(x.unique()),
        "writer": lambda x: list(x.unique())
    }).reset_index()

    return df_tolist

# 함수 정의: num_negative만큼 negative_sampling하기
def negative_sampling(df:pd.DataFrame,
                      user_col: str,
                      item_col: str,
                      num_negative: float
                      ) -> pd.DataFrame:
    """
    주어진 데이터프레임의 각 사용자에 대해 부정 샘플을 생성하고, 긍정 샘플과 결합하여 최종 데이터프레임을 반환합니다.

    Args:
        df (pd.DataFrame): user_col과 item_col을 column으로 갖는 데이터프레임
        user_col (str):  데이터프레임에서 사용자 ID를 나타내는 변수명
        item_col (str): 데이터프레임에서 아이템 ID를 나타내는 변수명
        num_negative (float): negative sample의 수

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
        pd.DataFrame: _description_
    """

    if 'review' in df.columns:
        positive_df =  df[df["review"]==1]
        pivot_count_df = positive_df[pivot_col].value_counts()
    
    else:
        pivot_count_df = df[pivot_col].value_counts()

    df[col_name] =  df[pivot_col].map(pivot_count_df)
    
    return df