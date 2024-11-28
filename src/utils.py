# src/utils.py

import math
import os
import random

import bottleneck as bn
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm


def set_seed(seed):
    """
    재현성을 위한 랜덤 시드를 설정하는 메서드
    다양한 라이브러리의 랜덤 시드를 설정 (Python의 random 모듈, Numpy, PyTorch(CPU 및 GPU))

    Args:
        seed (int): 랜덤 숫자 생성을 위한 시드 값
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_path(path: str) -> None:
    """
    디렉토리가 존재하는지 확인하고, 존재하지 않으면 생성하는 메서드

    Args:
        path (str): 확인하고 생성할 경로
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def recall_at_k(
        actual: list[int],
        predicted: list[int],
        topk: int
    ) -> float:
    """
    RECALL@K를 계산하는 메서드

    Args:
        actual (list[int]): 실제 값(ground truth) 리스트
        predicted (list[int]): 예측 값 리스트
        topk (int): 상위 K개 예측 항목의 수

    Returns:
        float: RECALL@K 값
    """
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])

        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1

    return sum_recall / true_users


def precision_at_k(
        actual: list[int],
        predicted: list[int],
        topk: int
    ) -> float:
    """
    PRECISION@K를 계산하는 메서드

    Args:
        actual (list[int]): 실제 값(ground truth) 리스트
        predicted (list[int]): 예측 값 리스트
        topk (int): 상위 K개 예측 항목의 수

    Returns:
        float: PRECISION@K 값
    """
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def apk(
        actual: list[int],
        predicted: list[int],
        topk: int
    ) -> float:
    """
    AP@K를 계산하는 메서드

    Args:
        actual (list[int]): 실제 값(ground truth) 리스트
        predicted (list[int]): 예측 값 리스트
        topk (int): 상위 K개 예측 항목의 수

    Returns:
        float: AP@K 값
    """
    if len(predicted) > topk:
        predicted = predicted[:topk]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), topk)


def mapk(
        actual: list[int],
        predicted: list[int],
        topk: int
    ) -> float:
    """
    MAP를 계산하는 메서드

    Args:
        actual (list[int]): 실제 값(ground truth) 리스트
        predicted (list[int]): 예측 값 리스트
        topk (int): 상위 K개 예측 항목의 수

    Returns:
        float: MAP 값
    """
    return np.mean([apk(a, p, topk) for a, p in zip(actual, predicted)])


def idcg_k(k: int) -> float:
    """
    IDCG@K를 계산하는 메서드

    Args:
        k (int): 상위 K개 예측 항목의 수

    Returns:
        float: IDCG@K 값
    """
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def ndcg_k(
        actual: list[int],
        predicted: list[int],
        topk: int
    ) -> float:
    """
    NDCG@K를 계산하는 메서드

    Args:
        actual (list[int]): 실제 값(ground truth) 리스트
        predicted (list[int]): 예측 값 리스트
        k (int): 상위 K개 예측 항목의 수

    Returns:
        float: NDCG@K 값
    """
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg

    return res / float(len(actual))


def ndcg_binary_at_k_batch(
        X_pred: np.ndarray,
        heldout_batch: csr_matrix,
        k: int
    ) -> np.ndarray:
    """
    배치에서 NDCG@K를 계산하는 메서드

    Args:
        X_pred (np.ndarray): 예측 값 행렬
        heldout_batch (csr_matrix): 실제 값(ground truth) user-item interaction matrix
        k (int): 상위 K개 예측 항목의 수

    Returns:
        np.ndarray: 각 사용자에 대한 NDCG 값의 배열
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                    idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                        idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                    for n in heldout_batch.getnnz(axis=1)])

    return DCG / IDCG


def recall_at_k_batch(
        X_pred: np.ndarray,
        heldout_batch: csr_matrix,
        k: int
    ) -> np.ndarray:
    """
    배치에서 RECALL@K를 계산하는 메서드

    Args:
        X_pred (np.ndarray): 예측 값 행렬
        heldout_batch (csr_matrix): 실제 값(ground truth) user-item interaction matrix
        k (int): 상위 K개 예측 항목의 수

    Returns:
        np.ndarray: 각 사용자에 대한 RECALL 값의 배열
    """
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))

    return recall


class EarlyStopping:
    """계산된 valid 평가지표가 patience 이후 개선되지 않는 경우 훈련 조기 중단"""

    def __init__(
            self,
            checkpoint_path: str,
            patience: int = 10,
            verbose: bool = False,
            delta: float = 0
        ):
        """
        Args:
            checkpoint_path (str): 모델 체크포인트를 저장할 경로
            patience (int): 마지막으로 검증 손실이 개선된 이후 기다릴 시간 (기본값: 10)
            verbose (bool): True일 경우, 각 검증 손실 개선에 대한 메시지를 출력 (기본값: False)
            delta (float): 개선으로 간주되기 위한 모니터링된 수치의 최소 변화량 (기본값: 0)
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score: list[float]) -> bool:
        """
        현재 점수를 이전 최고 점수와 비교하는 메서드

        Args:
            score (list[float]): 현재 점수 리스트

        Returns:
            bool: 개선되지 않았다면 True, 개선되었다면 False 반환
        """
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False

        return True

    def __call__(
            self,
            score: list[float],
            model: torch.nn.Module
        ):
        """
        EarlyStopping을 호출하여 현재 점수를 평가하고 모델을 저장하는 메서드

        Args:
            score (list[float]): 현재 점수 리스트
            model (torch.nn.Module): 저장할 모델
        """
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(
            self,
            score: list[float],
            model: torch.nn.Module
        ):
        """
        모델의 체크포인트를 저장하는 메서드

        Args:
            score (list[float]): 현재 점수 리스트
            model (torch.nn.Module): 저장할 모델
        """
        if self.verbose:
            print("Better performance. Saving model ...")
        if self.checkpoint_path.split('/')[1].split('_')[0] in ("EASE", "EASER"):
            directory, filename = os.path.split(self.checkpoint_path)
            name, _ = os.path.splitext(filename)
            new_file_path = os.path.join(directory, f"{name}.npy")
            np.save(new_file_path, model.B)
        else:
            torch.save(model.state_dict(), self.checkpoint_path)

        self.score_min = score


def save_recommendations(
        recommendations: list[list[float]],
        idx_to_user: dict[int, int],
        idx_to_item: dict[int, int],
        output_filename: str
    ):
    """
    추천 결과를 submission을 위한 양식에 맞게 바꾼 후, 파일로 저장하는 메서드

    Args:
        recommendations (list[list[float]]): 유저별 추천 아이템 리스트
        idx_to_user (dict[int, int]): 인덱스를 유저 ID로 매핑시키기 위한 딕셔너리
        idx_to_item (dict[int, int]): 인덱스를 아이템 ID로 매핑시키기 위한 딕셔너리
        output_filename (str): 저장할 경로 및 파일 이름
    """
    user_ids = []
    item_ids = []
    for user_idx, items in enumerate(recommendations):
        user_id = idx_to_user[user_idx]

        for item_idx in items:
            user_ids.append(user_id)
            item_ids.append(idx_to_item[item_idx])

    output_df = pd.DataFrame({'user': user_ids, 'item': item_ids})
    output_df.to_csv(output_filename, index=False)
    print(f"Recommendations saved to {output_filename}")


def ensemble_models(outputs: pd.DataFrame, p: list[int]) -> pd.DataFrame:
    """
    모델을 가중치만큼 랜덤 샘플링하여 앙상블하는 메서드

    Args:
        outputs (pd.DataFrame): 각 모델의 output을 이어붙인 데이터프레임
        p (list[int]): 각 모델의 가중치

    Returns:
        pd.DataFrame: 샘플링이 완료된 데이터프레임
    """
    output_group = outputs.groupby("user")
    sampled = pd.DataFrame(columns=["user", "item"])
    for user_id, group in tqdm(output_group, desc="Sampling..."):
        group = group.iloc[:, 1:]
        for i in range(len(p)):
            for _ in range(p[i]-1):
                group = pd.concat([group, group.iloc[:, i]], axis=1)

        items = group.values.flatten()

        lst = []
        for _ in range(10):
            r = np.random.choice(items)
            lst.append(r)
            items = items[items != r]

        result = pd.DataFrame({"user": user_id, "item": lst})
        sampled = pd.concat([sampled, result], axis=0)

    return sampled


def get_outputs(model_list: list[str], output_path: str) -> pd.DataFrame:
    """
    모델 이름을 포함한 리스트와 파일이 저장된 경로를 입력받아 ensemble_models 함수에 입력할 수 있는 형태로 변환하는 함수

    Args:
        model_list (list[str]): 모델의 이름을 담은 리스트
        output_path (str): csv 파일이 저장되어 있는 경로

    Returns:
        pd.DataFrame: 결과 데이터프레임
    """
    print("File Load:", *model_list)
    for i, model_name in enumerate(model_list):
        output = pd.read_csv(os.path.join(output_path, model_name) + "_output.csv")

        if i == 0:
            outputs = output
        else:
            outputs = pd.concat([outputs, output["item"]], axis=1)

    return outputs
