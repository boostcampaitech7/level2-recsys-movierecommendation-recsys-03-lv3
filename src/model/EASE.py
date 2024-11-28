# src/model/EASE.py

import numpy as np
from scipy.sparse import csr_matrix


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

    def loss_function_ease(self, X: csr_matrix) -> float:
        """
        EASE 계열 모델의 loss를 계산하는 메서드

        Args:
            X (csr_matrix): 사용자-아이템 희소 행렬 (학습 데이터)

        Returns:
            float: 계산된 loss 값
        """
        if self.B is None:
            raise ValueError("Model has not been trained yet.")

        X_dense = X.toarray()
        reconstruction_error = np.linalg.norm(X_dense - X_dense @ self.B, ord="fro")**2
        regularization_term = self.reg_lambda * np.linalg.norm(self.B, ord="fro")**2
        return reconstruction_error + regularization_term
