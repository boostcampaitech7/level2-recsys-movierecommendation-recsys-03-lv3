## 🌳 File Tree 🌳


```
📂 src
├── 📂 models
│   ├── __init__.py 
│   ├── DeepFM.py
│   ├── EASE.py
│   ├── EASER.py
│   └── MultiVAE.py
|
├── loader.py
├── preprocessing.py
├── trainer.py
├── utils.py
└── README.md
```

## Manual

### **model**

#### DeepFM.py

  DeepFM 모델을 구현한 파일
  
- **`__init__(self, input_dims, embedding_dim, mlp_dims: , drop_rate)`** : DeepFM 모델의 주요 구성요소(FM과 MLP)를 초기화하는 메서드
    - `input_dims` (list[int]) : 각 입력 차원의 크기 리스트
    - `embedding_dim` (int) : 임베딩 차원 크기
    - `mlp_dims` (list[int]) : MLP의 각 레이어 차원 리스트
    - `drop_rate` (float) : 드롭아웃 비율(default: 0.1)
      
- **`fm(self, x: torch.Tensor) -> torch.Tensor`** : Factorization Machine(FM)을 통해 1차 및 2차 상호작용 값을 계산하여 예측값 생성하는 메서드
    - `x` (torch.Tensor) : 입력 텐서
    - 반환값 : FM 계산 결과값 (**`torch.Tensor`**)
      
- **`mlp(self, x: torch.Tensor) -> torch.Tensor`** : Multi-Layer Perceptron(MLP)을 통해 비선형 변환을 학습하고 예측값 생성하는 메서드
    - `x` (torch.Tensor) : 입력 텐서
    - 반환값 : MLP 계산 결과값 (**`torch.Tensor`**)
      
- **`forward(self, x: torch.Tensor) -> torch.Tensor`** : 모델의 순전파 과정을 정의하여 최종 예측값 생성하는 메서드
    - `x` (torch.Tensor) : 입력 텐서
    - 반환값 : 최종 예측값으로, sigmoid 활성화 함수 적용됨 (**`torch.Tensor`**)
          
        
#### EASE.py

EASE(Efficient Automatic Scoring Estimation) 모델을 구현한 파일

- **`__init__(self, reg_lambda: float)`:** EASE 클래스의 초기화 메서드로, 정규화 계수 `reg_lambda`를 설정
    - `reg_lambda` : 정규화에 사용되는 하이퍼파라미터.
      
- **`train(self, X: csr_matrix) -> None`:** EASE 모델을 학습시키는 메서드
    - `X` : user-item interaction matrix로, CSR(Compressed Sparse Row) 형식의 행렬을 입력받습니다.
      
- **`predict(self, X: csr_matrix) -> np.ndarray`:** 학습된 모델을 사용해 예측값을 계산하는 메서드
    - `X` : user-item interaction matrix
    - 반환값 : 모델이 예측한 interaction 값 (**`np.ndarray`)**.
      
- **`loss_function_ease(self, X: csr_matrix) -> float`:** EASE 모델의 loss 값을 계산하는 메서드
    - `X` : user-item interaction matrix로, CSR 형식의 행렬을 입력받습니다.
    - 반환값 : reconstruction loss와 regularization term의 합 (**`float`**).
          
    
#### EASER.py

EASER(Extended EASE with Regularization and Smoothing) 모델을 구현한 파일

- **`__init__(self, reg_lambda: float, smoothing: float = 0.01)`:** EASER 클래스의 초기화 메서드
    - `reg_lambda` : 정규화에 사용되는 하이퍼파라미터
    - `smoothing` : smoothing 값을 설정하는 하이퍼파라미터 (기본값: 0.01)
- **`train(self, X: csr_matrix) -> None`:** EASER 모델을 학습시키는 메서드
    - `X` : user-item interaction matrix로, CSR(Compressed Sparse Row) 형식의 행렬을 입력받습니다.
- **`predict(self, X: csr_matrix) -> np.ndarray`:** 학습된 EASER 모델로 예측값을 계산하는 메서드
    - `X` : user-item interaction matrix로, CSR 형식의 행렬을 입력받습니다.
    - 반환값 : 모델이 예측한 interaction 값 (**`np.ndarray`**).
- **`loss_function_ease(self, X: csr_matrix) -> float`:** EASER 모델의 loss 값을 계산하는 메서드
    - `X` : user-item interaction matrix로, CSR 형식의 행렬을 입력받습니다.
    - 반환값 : reconstruction loss와 regularization term의 합 (**`float`**).

#### MultiVAE.py

MultiVAE(Variational AutoEncoder) 모델을 구현한 파일

- **`__init__(self, input_dim: int, hidden_dims: tuple[int, int] = (600, 200), dropout: float = 0.5)`:** MultiVAE 클래스의 초기화 메서드
    - `input_dim` : 입력 데이터의 차원
    - `hidden_dims` : 인코더와 디코더의 hidden layer 차원 수 (기본값: (600, 200))
    - `dropout` : 드롭아웃 비율 (기본값: 0.5)
    
- **`forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`:** MultiVAE 모델의 순전파 메서드
    - `x` : 입력 데이터 (user-item interaction matrix, **`torch.Tensor`**)
    - 반환값 :
        - `recon_x` : 재구성된 입력 데이터
        - `mean` : latent space의 평균
        - `logvar` : latent space의 로그 분산
- **`loss_function_multivae(self, x: torch.Tensor, recon_x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> float`:** MultiVAE 모델의 loss를 계산하는 메서드
    - `x` : 입력 데이터
    - `recon_x` : 재구성된 데이터
    - `mean` : latent space의 평균
    - `logvar` : latent space의 로그 분산
    - `beta` : KL divergence에 대한 가중치 (기본값: 1.0)
    - 반환값 : 재구성 손실(**`BCE`**)과 KL divergence 손실(**`KLD`**)의 가중 합으로 계산된 loss 값.
- **`init_weights(self)`:** 모델의 가중치를 초기화하는 메서드
    - Xavier 초기화와 정규 분포 기반으로 인코더 및 디코더의 가중치와 편향을 초기화.

#### dataset.py

PyTorch **`Dataset`** 클래스를 상속하여 추천 시스템 학습에 필요한 데이터를 효율적으로 관리하는 파일

- **`ContextDataset` 클래스** : PyTorch `Dataset`을 상속하여 컨텍스트 데이터를 효율적으로 관리하고, 학습 및 평가에 사용할 수 있도록 구성
    - **`__init__(self, X: torch.Tensor, y: torch.Tensor)`** :
        - 데이터셋의 특징(feature)인 `X`와 레이블(label)인 `y`를 초기화
        - 입력 데이터는 PyTorch 텐서로 변환되며, `X`와 `y`는 정수형(Long) 텐서로 저장
    - **`__len__(self)`** :
        - 데이터셋의 전체 크기를 반환 (`self.y`의 크기를 기준으로 함)
    - **`__getitem__(self, index: int)`** :
        - 지정된 인덱스에 해당하는 데이터를 반환
        - 레이블(`y`)이 존재하지 않을 경우, 특징(feature) 데이터 `X`만 반환
        - 레이블이 존재할 경우, `(X[index], y[index])`를 튜플 형태로 반환
    
#### loader.py

데이터 로드 및 전처리를 수행하며, 추천 시스템 학습 및 평가에 필요한 데이터를 준비하는 파일

- **`load_dataset(args: Namespace)`** : 데이터셋을 불러오고 전처리를 수행하여 학습에 사용할 데이터프레임을 생성
    1. `train_ratings`, `years`, `writers`, `titles`, `genres`, `directors` 등의 데이터를 불러온다.
    2. 장르, 감독, 작가의 상위 k개를 필터링하고 인코딩을 수행한다.
    3. 결측값 처리, 제목 정제, Negative Sampling을 적용한다.
    4. 중복된 `(user, item, time)` 데이터를 제거한다.
    - 반환값: 전처리 완료 후 병합된 데이터프레임
- **`Loader` 클래스** : 데이터 로드의 기본 틀을 제공하는 추상 클래스
- **`DeepFM` 클래스**
    - Loader를 상속하여, DeepFM 모델에 필요한 데이터를 준비
    - **`data_split(args: Namespace, data: pd.DataFrame)`** :
        - 유저별 데이터를 그룹화하여 학습/검증 세트로 나눔
        - 반환값: 분할된 학습/테스트 데이터프레임
    - **`data_loader(cat_features: list, batch_size: int, X_data: pd.DataFrame, y_data: pd.DataFrame, shuffle: bool)`** :
        - 데이터프레임을 텐서로 변환하고 배치 단위로 PyTorch DataLoader 객체를 생성
- **`EASE` 클래스**
    - 행렬 기반 추천 모델인 EASE에 적합한 데이터 준비를 수행
    - **`train_test_split(data: pd.DataFrame, split_ratio: float)`** :
        - 유저별 데이터를 시간순으로 정렬하고 학습 및 테스트 데이터로 분할
- **`EASER` 클래스**
    - **`EASE`** 클래스를 확장하며, 기본 로직을 재사용
- **`MultiVAE` 클래스**
    - **`EASE`** 클래스를 확장하여 MultiVAE 모델의 입력 데이터 구조를 준비
- **`ContextDataset` 클래스**
    - PyTorch `Dataset`을 상속하여 입력 데이터를 효율적으로 관리
    - **`__getitem__(index: int)`** :
        - 데이터의 특정 인덱스를 반환하며, 레이블(`y`) 유무에 따라 반환값이 달라짐

#### preprocessing.py

데이터 전처리를 위한 여러 가지 함수를 포함한 파일
  
- **`filter_top_k_by_count(df: pd.DataFrame, sel_col: str, pivot_col: str, top_k: int, ascending: bool = True) -> pd.DataFrame`** : 데이터프레임에서 특정 범주형 변수의 빈도를 계산하고, 각 그룹(피벗 컬럼 기준)에서 상위 k개의 범주만 추출
- **`label_encoding(df: pd.DataFrame, label_col: str, pivot_col: str = None) -> pd.DataFrame`** : 범주형 데이터를 수치형으로 변환하는 라벨 인코딩을 수행
- **`multi_hot_encoding(df: pd.DataFrame, label_col: str, pivot_col: str) -> pd.DataFrame`** : 범주형 데이터에 멀티-핫 인코딩을 적용하여, 각 범주를 이진 벡터로 변환
- **`preprocess_title(df: pd.DataFrame, col: str = "title") -> pd.DataFrame`** : 제목 텍스트를 정규 표현식을 사용하여 전처리
- **`fill_na(args: Namespace, df: pd.DataFrame, col: str) -> pd.DataFrame`** : 결측치를 처리하는 함수로, 특정 열에 대해 결측치를 채우는 방법을 정의
- **`negative_sampling(df: pd.DataFrmae, user_col: str, item_dol: str, num_negative: int, na_list: list[str] = ["title", "year", "genre", "director", "writer"]) -> pd.DataFrame`** : 데이터프레임에서 각 사용자에 대해 부정 샘플(negative sample)을 생성하고, 기존의 긍정 샘플(positive sample)과 결합하여 최종 데이터프레임을 반환
- **`pivot_count(df: pd.DataFrame, pivot_col: str, col_name: str) -> pd.DataFrame`** : 데이터프레임에서 특정 열(pivot_col)의 값을 카운팅하여, 그 결과를 새로운 열(col_name)로 추가
- **`merge_dataset(title: pd.DataFrame, years: pd.DataFrame, genres: pd.DataFrame, directors: pd.DataFrame, writers: pd.DataFrame) -> pd.DataFrame`** : 여러 개의 데이터프레임(제목, 개봉 연도, 장르, 감독, 작가)을 하나의 아이템 데이터프레임으로 병합
- **`id2idx(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], int, int]`** : 사용자와 아이템 ID를 0부터 시작하는 고유 인덱스로 매핑
- **`idx2id(user_to_idx: dict[int, int], item_to_idx: dict[int, int]) -> tuple[dict[int, int], dict[int, int]]`** : 고유 인덱스를 사용자와 아이템 ID로 다시 매핑
- **`df2ma(data: pd.DataFrame, num_users: int, num_items: int) -> csr_matrix`** : 주어진 데이터프레임에서 사용자와 아이템의 상호작용 정보를 추출하여, 희소 행렬(sparse matrix) 형태의 사용자-아이템 상호작용 행렬을 생성
    
#### trainer.py
- **Trainer 클래스**

  머신러닝 모델을 학습하고 평가하는 데 필요한 기능을 제공
  
    - **`__init__(self, model: nn.Module, train_dataloader: Union[DataLoader, csr_matrix], eval_dataloader: Union[DataLoader, csr_matrix], submission_dataloader: Union[DataLoader, csr_matrix], seen_items: Union[set, csr_matrix], args: Namespace)`**:
      모델과 데이터 로더를 초기화하고, CUDA 사용 가능 여부를 확인하여 모델을 GPU로 이동시킵니다. 또한, 모델 이름에 따라 최적화기를 설정
    - **`train(self, epoch)`**: 주어진 epoch 동안 모델을 훈련
    - **`valid(self, epoch)`**: 검증 데이터를 사용하여 모델을 평가
    - **`test(self, epoch)`**: 테스트 데이터를 사용하여 모델을 평가
    - **`submission(self, epoch)`**: 제출 데이터를 사용하여 모델을 평가
    - **`iteration(self, epoch, dataloader, mode="train")`**: 각 epoch 동안의 훈련, 검증, 테스트 또는 제출을 위한 반복을 수행하는 메서드
    - **`get_full_sort_score(self, epoch, actual, predicted)`**: 여러 개의 랭킹 평가지표를 계산
    - **`save(self, file_name: str)`**: 모델의 상태(가중치 및 편향 등)를 파일에 저장
    - **`load(self, file_name: str)`**: 저장된 모델의 상태를 파일에서 불러옴
- **DeepFM 클래스**

  Trainer 클래스를 상속받아, DeepFM 모델을 훈련하고 평가하는 기능을 구현
  
    - **`__init__(self, model, train_dataloader, eval_dataloader, submission_dataloader, seen_items, args)`**:
      부모 클래스인 Trainer의 초기화 메서드를 호출하여 모델과 데이터 로더를 설정
    - **`iteration(self, epoch, dataloader, mode="train")`**: 메서드는 주어진 epoch 동안 모델을 훈련, 검증 또는 테스트하는 과정을 수행
- **EASE 클래스**

  Trainer 클래스를 상속받아, EASE (Embarrassingly Shallow Autoencoder) 모델을 훈련하고 평가하는 기능을 구현
  
    - **`__init__(self, model, train_dataloader, eval_dataloader, submission_dataloader, seen_items, args)`**:
      부모 클래스인 Trainer의 초기화 메서드를 호출하여 모델과 데이터 로더를 설정
    - **`iteration(self, _, data, mode="train")`**: 메서드는 주어진 데이터와 모드에 따라 모델을 훈련, 검증 또는 테스트하는 과정을 수행
- **EASER 클래스**

  EASE 클래스를 상속받아 동일한 기능을 제공
  
    - **`__init__(self, model, train_dataloader, eval_dataloader, submission_dataloader, seen_items, args)`**:
      EASER 클래스의 초기화 메서드는 부모 클래스인 EASE의 초기화 메서드를 호출하여 모델과 데이터 로더를 설정
- **MultiVAE 클래스**
    - Trainer 클래스를 상속받아 Variational Autoencoder (VAE) 기반의 추천 시스템을 구현하는 클래스
    - **`__init__(self, model, train_dataloader, eval_dataloader, submission_dataloader, seen_items, args)`**:
        - MultiVAE 클래스의 초기화 메서드는 부모 클래스인 Trainer의 초기화 메서드를 호출하여 모델과 데이터 로더를 설정
    - **`iteration(self, epoch, data, mode="train")`**:
        - 모델의 훈련, 검증, 또는 예측을 수행
    
#### utils.py
- **`set_seed(seed)`** : 재현성을 위해 다양한 라이브러리의 랜덤 시드 설정하는 메서드
- **`check_path(path)`** : 디렉토리가 존재하는지 확인하고, 존재하지 않으면 생성하는 메서드
- **`recall_at_k(actual, predicted, topk) -> float`** : RECALL@K를 계산하는 메서드
- **`precision_at_k(actual, predicted, topk) -> float`** : PRECISION@K를 계산하는 메서드
- **`apk(actual, predicted, topk) -> float`** : AP@K를 계산하는 메서드
- **`mapk(actual, predicted, topk) -> float`** : MAP를 계산하는 메서드
- **`idch_k(k) -> float`** : IDCG@K를 계산하는 메서드
- **`ndcg_k(actual, predicted, topk) -> float`** : NDCG@K를 계산하는 메서드
- **`ndcg_binary_at_k_batch(X_pred, heldout_batch, k) -> np.ndarray`** : 배치에서 NDCG@K를 계산하는 메서드
- **`recall_at_k_batch(X_pred, heldout_batch, k) -> np.ndarray`** : 배치에서 RECALL@K를 계산하는 메서드
- **`EarlyStopping`** 클래스 :
    - 계산된 valid 평가지표가 patience 이후 개선되지 않는 경우 훈련을 조기 종료합니다
    - **`init(self, checkpoint_path, patience, verbose, delta)`** : 검증 점수가 일정 기간 동안 개선되지 않으면 훈련을 조기 종료하기 위한 초기화 설정
    - **`compare(score)`**: 현재 점수와 최고 점수를 비교하는 메서드
    - **`__call__(score, model)`**: EarlyStopping을 호출하여 현재 점수를 평가하고 모델을 저장하는 메서드
    - **`save_checkpoint(score, model)`**: 모델의 체크포인트 저장하는 메서드
- **`save_recommendations(recommendations, idx_to_user, idx_to_item, output_filename)`** : 추천 결과를 submission을 위한 양식에 맞게 바꾼 후, 파일로 저장하는 메서드
- **`ensemble_models(outputs, p) -> pd.DataFrame`** : 모델을 가중치만큼 랜덤 샘플링하여 앙상블하는 메서드
- **`get_outputs(model_list, output_path) -> pd.DataFrame`** : 모델 이름을 포함한 리스트와 파일이 저장된 경로를 입력받아 `ensemble_models` 함수에 입력할 수 있는 형태로 변환하는 함수
