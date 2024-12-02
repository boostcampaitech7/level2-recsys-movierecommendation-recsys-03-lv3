<h1 align="center"><a href='https://ginger-scion-7bd.notion.site/13bb563074a68145a3bbc81c40f1f0df?v=13bb563074a68172a5ec000cacc5fa79&pvs=4'>RecSys-03 ㄱ해줘</a></h1>
<br></br>

## 🏆 대회 개요 🏆

MovieLens 데이터를 전처리하여 competition 용도로 재구성한 데이터를 활용한다. sequence를 바탕으로 마지막 item만을 예측하는 sequential recommendation 시나리오와 비교하여, 보다 복잡하며 실제와 비슷한 상황을 가정한다.

- Objective : 
  **사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측**
- 평가 지표 : (normalized) **Recall@K**

<br></br>
## 👨‍👩‍👧‍👦 팀 소개 👨‍👩‍👧‍👦
    
|강성택|김다빈|김윤경|김희수|노근서|박영균|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<a href='https://github.com/TaroSin'><img src='https://github.com/user-attachments/assets/75682bd3-bcff-433e-8fe5-6515a72361d6' width='200px'/></a>|<a href='https://github.com/BinnieKim'><img src='https://github.com/user-attachments/assets/ff639e97-91c9-47e1-a0c8-a5fc09c025a6' width='200px'/></a>|<a href='https://github.com/luck-kyv'><img src='https://github.com/user-attachments/assets/015ec963-d1b4-4365-91c2-d513e94c2b8a' width='200px'/></a>|<a href='https://github.com/0k8h2s5'><img src='https://github.com/user-attachments/assets/526dc87c-0122-4829-8e94-bce6f15fc068' width='200px'/></a>|<a href='https://github.com/geunsseo'><img src='https://github.com/user-attachments/assets/0a1a27c1-4c91-4fdf-b350-1540c835ee72' width='200px'/></a>|<a href='https://github.com/0-virus'><img src='https://github.com/user-attachments/assets/98470105-260e-443d-8592-c139d7918b5e' width='200px'/></a>|

<br></br>

## 🌳 File Tree 🌳

```
{level2-recsys-movierecommendation-recsys-03-lv3}
│
├── 📁 config
│   └── config_baseline.yaml
│
├── 📂 docs
│   └── Movie Recommendation_RecSys_팀리포트(03조).pdf
│
├── 📂 RecBole
│   ├── 📂 config
│   │   └──recbole_config.yaml
│   │
│   ├── recbole_inference.py
│   └── recbole_main.py
│
├── 📂 EDA
│   ├── davin_EDA.ipynb
│   ├── gs_EDA.ipynb
│   ├── hs_EDA.ipynb
│   ├── tarosin_EDA.ipynb
│   ├── yoon_EDA.ipynb
│   └── zv_EDA.ipynb
│
├── 📂 src
│   ├── 📂 model
│   │   ├── __init__.py
│   │   ├── DeepFM.py
│   │   ├── EASE.py
│   │   ├── EASER.py
│   │   └── MultiVAE.py
│   │
│   ├── loader.py
│   ├── preprocessing.py
│   ├── trainer.py
│   └── utils.py
│
├── .gitignore
├── ensemble.py
├── inference.py
├── main.py
├── requirements.txt
├── run_catboost.py
└── README.md
```

<br></br>

## ▶️ 실행 방법 ▶️

- Package install
    
    ```bash
    pip install -r requirements.txt
    ```
    
- Model training
    
    ```bash
    # main.py 실행
    python main.py --m DeepFM   --e 100 --r baseline --d cuda
    python main.py --m EASE     --e 100 --r baseline --d cpu
    python main.py --m EASER    --e 100 --r baseline --d cpu
    python main.py --m MultiVAE --e 100 --r baseline --d cuda
    
    # run_catboost.py 실행 (inference 포함)
    python run_catboost.py
    ```

- Inference & Ensemble

    ```bash
    # inference.py 실행
    python inference.py --m DeepFM   --r baseline --d cuda
    python inference.py --m EASE     --r baseline --d cpu
    python inference.py --m EASER    --r baseline --d cpu
    python inference.py --m MultiVAE --r baseline --d cuda

    # ensemble.py 실행
    python ensemble.py --m EASER CatBoost MultiVAE --w 7 2 1 --o output/
    ```

## 🥇 Result 🥇
#### 결과 -> 8등
![image](https://github.com/user-attachments/assets/07572397-f5f5-4557-afe3-ea8375d4b843)



#### 제출 1 - EASER 모델

![image](https://github.com/user-attachments/assets/eb5797fd-4f2d-4c61-b118-7397224ac108)


#### 제출 2 - EASE 모델

![image](https://github.com/user-attachments/assets/4a9a0277-2f4f-403f-8f02-072072628564)



<br></br>
## GitHub Convention


- ***main*** branch는 배포이력을 관리하기 위해 사용,

  ***movie*** branch는 기능 개발을 위한 branch들을 병합(merge)하기 위해 사용
- 모든 기능이 추가되고 버그가 수정되어 배포 가능한 안정적인 상태라면 ***movie*** branch에 병합(merge)
- 작업을 할 때에는 개인의 branch를 통해 작업
- EDA
    
    branch명 형식은 “**EDA-name**” 으로 작성 ex) EDA-TaroSin
    
    파일명 형식은 “**name_EDA**” 으로 작성 ex) TaroSin_EDA
    
- 데이터 전처리팀 branch 관리 규칙
    
    ```
    movie
    └── data
        ├── data-loader         # 데이터셋 로드, 데이터 로더 개발
        └── data-preprocessing  # 전처리 및 피쳐 엔지니어링 개발
    

    ```
    
- 모델팀 branch 관리 규칙
    
    ```
   movie
    └── model
          ├── model-DeepFM     # 모델 개발 branch
          ├── model-general    # CF, 트리 모델 개발 branch
          ├── ...
          └── model-ensemble   # 앙상블 개발 branch
    ```
    
- *master(main)* branch에 Pull request를 하는 것이 아닌,
    
    ***data*** branch 또는 ***model*** branch에 Pull request 요청
    
- commit message는 아래와 같이 구분해서 작성 (한글)

  ex) git commit -m “**docs**: {내용} 문서 작성”
  
  ex) git commit -m “**feat**: {내용} 추가”
  
  ex) git commit -m “**fix**: {내용} 수정”
  
  ex) git commit -m “**test**: {내용} 테스트”

- Pull request merge 담당자 : **data - 근서** / **model - 성택** / **최종 - 영균**

  나머지는 ***movie*** branch 건드리지 말 것!

  merge commit message는 아래와 같이 작성

  ex) “**merge**: {내용} 병합“
- **Issues**, **Pull request**는 Template에 맞추어 작성 (커스텀 Labels 사용)
  
  Issues → 작업 → PR 순으로 진행

<br></br>

## Code Convention
PEP-8을 기반으로 일부 수정해서 사용
- 문자열을 처리할 때는 작은 따옴표를 사용하도록 합니다.
- 클래스명은 `카멜케이스(CamelCase)` 로 작성합니다. </br>
  함수명, 변수명은 `스네이크케이스(snake_case)`로 작성합니다.
- 객체의 이름은 해당 객체의 기능을 잘 설명하는 것으로 정합니다.  
    ```python
    # bad
    a = ~~~
    # good
    lgbm_pred_y = ~~~
    ```
- 상수는 대문자와 밑줄(`_`)로 작성하여, 변수와 구분되도록합니다.
    ```python
    MAX_CONNECTIONS = 100
    DEFAULT_TIMEOUT = 60
    ```
- 가독성을 위해 한 줄에 하나의 문장만 작성합니다.
- 들여쓰기는 4 Space 대신 Tab을 사용합시다.
- 주석은 설명하려는 구문에 맞춰 들여쓰기, 코드 위에 작성합니다.
    ```python
    # good
    def some_function():
      ...
    
      # statement에 관한 주석
      statements
    ```
    
- 키워드 인수를 나타낼 때나 주석이 없는 함수 매개변수의 기본값을 나타낼 때 기호 주위에 공백을 사용하지 마세요.
    
    ```python
    # bad
    def complex(real, imag = 0.0):
        return magic(r = real, i = imag)
    # good
    def complex(real, imag=0.0):
        return magic(r=real, i=imag)
    ```

- 연산자 사이에는 공백을 추가하여 가독성을 높입니다.
    
    ```python
    a+b+c+d # bad
    a + b + c + d # good
    ```
    
- (`_`), (`:`), (`;`) 등 구분자 다음에 값이 올 경우 공백을 추가하여 가독성을 높입니다.
    
    ```python
    arr = [1,2,3,4] # bad
    arr = [1, 2, 3, 4] # good
    ```
- 한 줄의 코드가 너무 길어질 경우, 괄호를 사용하여 줄을 나누어 작성합니다.
    ```python
    # 괄호를 사용한 예
    long_variable = (
        var_one
        + var_two
        - var_three
    )
    ```
- 클래스와 최상위 함수 정의 사이에는 빈 줄 2개를 사용합니다. 클래스 내부의 메서드들 사이에는 빈 줄 1개를 사용하여 가독성을 높입니다.
    ```python
    # 최상위 함수와 클래스 사이에 빈 줄 2개
    def example_function():
        pass
    
    
    class MyClass:
        def first_method(self):
            pass
    
        def second_method(self):
            pass
    ```
- 딕셔너리 항목의 경우, 항목을 줄마다 하나씩 나열하고 마지막 항목 뒤에 쉼표를 추가합니다.
    ```python
    my_dict = {
      "name": "Alice",
      "age": 30,
      "city": "New York",
    }
    ```
- 예외를 처리할 때는 메시지를 통해 어떤 예외가 발생했는지 알 수 있도록 합니다.
    ```python
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        print(f"Error occurred: {e}")
    ```
- 조건문에서 싱글톤 객체 (예: `None`, `True`, `False`) 를 비교할 때는 `==` 대신 `is` 를 사용합니다.
    ```python
    if my_var is None:
      print("my_var is None")
    ```
- 프라이빗(private) 변수나 메서드는 이름 앞에 밑줄 하나를 붙여 표현
    ```python
    class MyClass:
        def __init__(self):
            self._private_variable = 42
        
        def _private_method(self):
            pass
    ```
- 임포트는 항상 파일 상단에 위치시키고, 표준 라이브러리, 서드 파티 라이브러리, 로컬 모듈 순서로 정리합니다. 한 줄에 하나의 라이브러리만 임포트하며, 여러 모듈을 한 줄에 나열하지 않습니다.
  ```python
  import os
  import sys
  
  import numpy as np
  
  from my_local_module import my_function 
  ```
- 사용하지 않는 변수를 남기지 않고 제거하여 코드의 깔끔함을 유지합니다.
