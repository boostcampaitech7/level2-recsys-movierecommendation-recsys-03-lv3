import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

# MultiVAE 클래스 정의
class MultiVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=(600, 200), dropout=0.5):
        super(MultiVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1] * 2)  # mean과 logvar 생성
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.dropout(x)
        h = self.encoder(h)
        mean, logvar = torch.chunk(h, 2, dim=-1)  # 분리
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mean
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def loss_function(self, recon_x, x, mean, logvar, beta=1.0):
        BCE = -(x * torch.log(recon_x + 1e-10)).sum(dim=-1).mean()
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1).mean()
        return BCE + beta * KLD

# MultiVAE 학습 함수
def train_multivae(model, train_data, epochs=50, batch_size=128, lr=0.001, beta=1.0, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)  # 모델을 지정된 디바이스로 이동
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for start in range(0, train_data.shape[0], batch_size):
            end = min(start + batch_size, train_data.shape[0])
            batch = torch.FloatTensor(train_data[start:end].toarray()).to(device)

            optimizer.zero_grad()
            recon_batch, mean, logvar = model(batch)
            loss = model.loss_function(recon_batch, batch, mean, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / train_data.shape[0]:.4f}")

# 메인 함수
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

    # MultiVAE 모델 생성 및 학습
    input_dim = num_items
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU가 사용 가능하면 사용
    multivae_model = MultiVAE(input_dim)
    train_multivae(multivae_model, interaction_matrix, device=device)

    # 예측
    multivae_model.eval()
    with torch.no_grad():
        interaction_matrix_tensor = torch.FloatTensor(interaction_matrix.toarray()).to(device)
        predictions, _, _ = multivae_model(interaction_matrix_tensor)

    # 평가된 아이템 제외
    predictions[interaction_matrix.nonzero()] = -np.inf

    # 상위 N개 아이템 추천
    N = 10
    top_items_per_user = predictions.topk(N, dim=1)[1].cpu().numpy()  # GPU 사용 시 CPU로 이동

    # 인덱스 -> 실제 아이템 ID 변환
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    recommendations = [
        (unique_users[user_idx], idx_to_item[item_idx])
        for user_idx, items in enumerate(top_items_per_user)
        for item_idx in items
    ]

    # 결과 저장
    output_df = pd.DataFrame(recommendations, columns=['user', 'item'])
    output_df.to_csv("multivae.csv", index=False)
    print("Recommendations saved to multivae.csv")

if __name__ == "__main__":
    main()
