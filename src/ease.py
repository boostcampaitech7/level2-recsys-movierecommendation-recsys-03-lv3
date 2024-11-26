import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# EASE 클래스 정의
class EASE:
    def __init__(self, reg_lambda):
        self.B = None
        self.reg_lambda = reg_lambda

    def train(self, X):
        try:
            X_dense = X.toarray()
            G = X_dense.T @ X_dense
            diag_indices = np.diag_indices_from(G)
            G[diag_indices] += self.reg_lambda
            P = np.linalg.pinv(G)
            self.B = P / -np.diag(P)
            self.B[diag_indices] = 0
        except Exception as e:
            print(f"Error during training: {e}")

    def predict(self, X):
        X_dense = X.toarray() if isinstance(X, csr_matrix) else X
        return X_dense @ self.B

def main():
    # 데이터 로드
    data_path = "/data/ephemeral/home/movie/data/train/train_ratings.csv"
    ratings = pd.read_csv(data_path)

    # user와 item 고유 인덱스 매핑
    unique_users = ratings['user'].unique()
    unique_items = ratings['item'].unique()

    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

    # 인덱스 변환
    ratings['user'] = ratings['user'].map(user_to_idx)
    ratings['item'] = ratings['item'].map(item_to_idx)

    num_users = len(user_to_idx)
    num_items = len(item_to_idx)

    # 희소 행렬 생성
    rows, cols = ratings['user'].values, ratings['item'].values
    data = np.ones(len(ratings))
    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    # EASE 모델 학습
    reg_lambda = 500
    ease_model = EASE(reg_lambda)
    ease_model.train(interaction_matrix)

    # 예측
    predictions = ease_model.predict(interaction_matrix)

    # 평가된 아이템 제외
    predictions[interaction_matrix.nonzero()] = -np.inf

    # 상위 N개 아이템 추천
    N = 10
    top_items_per_user = []
    for user_idx in range(predictions.shape[0]):
        top_items = np.argsort(predictions[user_idx])[-N:][::-1]
        top_items_per_user.append(top_items)

    # 인덱스 -> 실제 아이템 ID 변환
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    recommendations = [
        (unique_users[user_idx], idx_to_item[item_idx])
        for user_idx, items in enumerate(top_items_per_user)
        for item_idx in items
    ]

    # 결과 저장
    output_df = pd.DataFrame(recommendations, columns=['user', 'item'])
    output_df.to_csv("ease.csv", index=False)
    print("Recommendations saved to ease.csv")

if __name__ == "__main__":
    main()
