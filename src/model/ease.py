# src/model/ease.py

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
