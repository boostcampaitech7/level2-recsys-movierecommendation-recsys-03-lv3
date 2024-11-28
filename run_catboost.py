# 필요한 라이브러리 및 모듈 임포트
import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Any, Dict, List

from src.loader import load_dataset

# 1. 데이터 로드 및 전처리
args: argparse.Namespace = argparse.Namespace(
    data_path="/data/ephemeral/home/movie/data/train",
    preprocessing=argparse.Namespace(
        ascending=False,
        encoding="MH",
        is_array=True,
        tree2array=False,
        negative_sampling=False,
    )
)

# 데이터 로드
data: pd.DataFrame = load_dataset(args)

# 필요 없는 열 제거 및 데이터 전처리
data = data.drop(columns=["time", "title"])
data["year_bin"] = (data["year"] // 5) * 5
data = data.drop(columns=["year"])
data["director"] = data["director"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else -1)


# 2. Negative Sampling
def negative_sampling(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    num_negative: int,
    na_list: List[str]
) -> pd.DataFrame:
    """
    Negative Sampling을 수행하여 Positive와 Negative 데이터 결합.

    Args:
        df (pd.DataFrame): 원본 데이터프레임.
        user_col (str): 사용자 컬럼 이름.
        item_col (str): 아이템 컬럼 이름.
        num_negative (int): 사용자당 생성할 Negative Sample 수.
        na_list (List[str]): 결측치를 확인 및 처리할 열 이름 목록.

    Returns:
        pd.DataFrame: Negative Sample이 추가된 데이터프레임.
    """
    # 전체 아이템 집합 및 사용자별 본 아이템 생성
    items: set = set(df[item_col].unique())
    user_items_dict: Dict[str, set] = df.groupby(user_col)[item_col].apply(set).to_dict()
    item_na_map: Dict[str, Dict[Any, Any]] = {
        col: df.groupby(item_col)[col].first().to_dict() for col in na_list
    }
    neg_samples: List[Dict[str, Any]] = []

    for user, u_items in tqdm(user_items_dict.items()):
        negative_items: List[int] = np.random.choice(
            list(items - u_items),
            min(num_negative, len(items - u_items)),
            replace=False
        )
        for item in negative_items:
            neg_sample: Dict[str, Any] = {user_col: user, item_col: item, "review": 0}

            for na_col in na_list:
                neg_sample[na_col] = item_na_map[na_col].get(item, None)

            neg_samples.append(neg_sample)

    neg_samples_df: pd.DataFrame = pd.DataFrame(neg_samples)
    raw_rating_df: pd.DataFrame = pd.concat([df, neg_samples_df], axis=0, ignore_index=True)
    raw_rating_df["review"] = raw_rating_df["review"].fillna(1).astype("int64")

    return raw_rating_df


na_columns: List[str] = ["Action", "Adventure", "Animation", "Children",
                         "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                         "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
                         "War", "Western", "director", "num_reviews_item", "year_bin"]
data = negative_sampling(data, user_col="user", item_col="item", num_negative=500, na_list=na_columns)


# 3. Train/Test Split
train: pd.DataFrame
test: pd.DataFrame
train, test = train_test_split(data, test_size=0.2, random_state=42)

# 4. CatBoost 모델 학습
X_train: pd.DataFrame = train.drop(columns=["review"])
y_train: pd.Series = train["review"]
X_test: pd.DataFrame = test.drop(columns=["review"])
y_test: pd.Series = test["review"]

categorical_features: List[str] = ["user", "item", "director", "year_bin"]

model: CatBoostClassifier = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    cat_features=categorical_features,
    verbose=100,
)

train_pool: Pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool: Pool = Pool(X_test, y_test, cat_features=categorical_features)

model.fit(train_pool, eval_set=test_pool)


# 5. 추천 생성
data_unique: pd.DataFrame = data.drop_duplicates(subset="item", keep="first")
metadata: Dict[int, Dict[str, Any]] = data_unique.set_index("item").drop(columns=["review", "user"]).to_dict(orient="index")


def generate_recommendations(model: CatBoostClassifier, user_ids: List[int], k: int = 10) -> pd.DataFrame:
    """
    사용자별 Top-K 추천 리스트 생성.

    Args:
        model (CatBoostClassifier): 학습된 CatBoost 분류기.
        user_ids (List[int]): 추천을 생성할 사용자 ID 목록.
        k (int): 추천할 상위 아이템 개수.

    Returns:
        pd.DataFrame: 사용자별 추천 결과 데이터프레임.
    """
    recommendations: List[Dict[str, Any]] = []
    all_items: set = set(data["item"].unique())
    user_seen_items: Dict[int, set] = train.groupby("user")["item"].apply(set).to_dict()

    for user in tqdm(user_ids):
        seen_items: set = user_seen_items.get(user, set())
        candidate_items: List[int] = list(all_items - seen_items)
        candidate_data: pd.DataFrame = pd.DataFrame([
            {"user": user, "item": item, **metadata[item]} for item in candidate_items
        ])
        candidate_data = candidate_data[X_train.columns]
        candidate_data["score"] = model.predict_proba(candidate_data)[:, 1]
        top_k_items: pd.Series = candidate_data.nlargest(k, "score")["item"]

        for item in top_k_items:
            recommendations.append({"user": user, "item": item})

    return pd.DataFrame(recommendations)


user_ids: np.ndarray = test["user"].unique()
recommendations: pd.DataFrame = generate_recommendations(model, user_ids, k=10)


# 6. 평가 지표 계산
def recall_at_k(recommendations: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> float:
    """
    Recall@K 계산.

    Args:
        recommendations (pd.DataFrame): 추천 결과.
        ground_truth (pd.DataFrame): 실제 정답 데이터.
        k (int): 상위 K개 아이템 고려.

    Returns:
        float: Recall@K 값.
    """
    hits: int = 0
    total: int = 0
    for user, rec_items in recommendations.groupby("user")["item"]:
        true_items = ground_truth[ground_truth["user"] == user]["item"].values
        hits += len(set(rec_items[:k]) & set(true_items))
        total += len(true_items)

    return hits / total if total > 0 else 0


def ndcg_at_k(recommendations: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> float:
    """
    NDCG@K 계산.

    Args:
        recommendations (pd.DataFrame): 추천 결과.
        ground_truth (pd.DataFrame): 실제 정답 데이터.
        k (int): 상위 K개 아이템 고려.

    Returns:
        float: NDCG@K 값.
    """
    total_ndcg: float = 0
    total_users: int = 0
    for user, rec_items in recommendations.groupby("user")["item"]:
        true_items = ground_truth[ground_truth["user"] == user]["item"].values
        dcg = sum([1 / np.log2(idx + 2) for idx, item in enumerate(rec_items[:k]) if item in true_items])
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(true_items), k))])
        total_ndcg += dcg / idcg if idcg > 0 else 0
        total_users += 1

    return total_ndcg / total_users if total_users > 0 else 0


ground_truth: pd.DataFrame = test[test["review"] == 1]
recall: float = recall_at_k(recommendations, ground_truth, k=10)
ndcg: float = ndcg_at_k(recommendations, ground_truth, k=10)

print(f"Recall@10: {recall:.4f}")
print(f"NDCG@10: {ndcg:.4f}")


# 7. 결과 저장
recommendations_sorted: pd.DataFrame = recommendations.sort_values(by="user").reset_index(drop=True)
recommendations_sorted.to_csv("submission.csv", index=False)
