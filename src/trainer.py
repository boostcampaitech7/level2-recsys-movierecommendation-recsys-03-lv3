# src/trainer.py

import os
from argparse import Namespace
from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix

from .utils import (
    recall_at_k,
    precision_at_k,
    mapk,
    ndcg_k,
    ndcg_binary_at_k_batch,
    recall_at_k_batch,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Union[DataLoader, csr_matrix],
        eval_dataloader: Union[DataLoader, csr_matrix],
        submission_dataloader: Union[DataLoader, csr_matrix],
        seen_items: Union[set, csr_matrix],
        args: Namespace,
    ):
        """
        Trainer 클래스 초기화 메서드

        Args:
            model (nn.Module): 학습할 모델
            train_dataloader (Union[DataLoader, csr_matrix]): train DataLoader 또는 train user-item interaction matrix
            eval_dataloader (Union[DataLoader, csr_matrix]): valid DataLoader 또는 valid user-item interaction matrix
            submission_dataloader (Union[DataLoader, csr_matrix]): submission DataLoader 또는 submission user-item interaction matrix
            seen_items (Union[set, csr_matrix]): 이미 본 item set 또는 train user-item interaction matrix
            args (Namespace): parser.parse_args()에서 반횐된 Namespace 객체
        """

        self.args = args
        self.model = model
        self.device = args.device
        if self.device == "cuda":
            self.cuda_condition = torch.cuda.is_available()
            if self.cuda_condition:
                self.model.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.submission_dataloader = submission_dataloader
        self.seen_items = seen_items

        if args.model_name not in ("EASE", "EASER"):
            self.optim = Adam(
                self.model.parameters(),
                lr=0.01,
            )

    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(
            self,
            epoch: int,
            actual: list[int],
            predicted: list[int],
        ) -> None:
        """
        여러 Ranking 평가지표를 한 번에 계산하는 메서드
        (RECALL@5, RECALL@10, PRECISION@10, MAP@10, NDCG@10)

        Args:
            epoch (int): 현재 epoch
            actual (list[int]): 실제 값(ground truth) 리스트
            predicted (list[int]): 예측 값 리스트
        """
        recall_5 = recall_at_k(actual, predicted, topk=5)
        recall_10 = recall_at_k(actual, predicted, topk=10)
        precision_10 = precision_at_k(actual, predicted, topk=10)
        map_10 = mapk(actual, predicted, k=10)
        ndcg_10 = ndcg_k(actual, predicted, topk=10)

        post_fix = {
            "RECALL@5": "{:.4f}".format(recall_5),
            "RECALL@10": "{:.4f}".format(recall_10),
            "PRECISION@10": "{:.4f}".format(precision_10),
            "MAP@10": "{:.4f}".format(map_10),
            "NDCG@10": "{:.4f}".format(ndcg_10),
        }
        print(f"[{epoch} Epoch]")
        print(post_fix)
        wandb.log(post_fix)

    def save(self, file_name: str) -> None:
        """
        모델의 상태(가중치, 편향 등)를 파일에 저장하는 메서드
        - EASE 또는 EASER 모델인 경우, 모델의 B 메트릭스를 .npy 파일로 저장
        - 그 외의 경우, 모델의 state_dict를 .pt 파일로 저장

        Args:
            file_name (str): 저장할 파일의 경로 및 이름
        """
        if self.args.model_name in ("EASE", "EASER"):
            directory, filename = os.path.split(file_name)
            name, _ = os.path.splitext(filename)
            new_file_path = os.path.join(directory, f"{name}.npy")
            np.save(new_file_path, self.model.B)
        else:
            torch.save(self.model.cpu().state_dict(), file_name)
            self.model.to(self.device)

    def load(self, file_name: str) -> None:
        """
        저장된 모델의 상태(가중치, 편향 등)를 파일에서 불러오는 메서드
        - EASE 또는 EASER 모델인 경우, .npy 파일에서 B 메트릭스 로드
        - 그 외의 경우, .pt 파일에서 모델의 state_dict를 로드

        Args:
            file_name (str): 로드할 파일의 경로 및 이름
        """
        if self.args.model_name in ("EASE", "EASER"):
            directory, filename = os.path.split(file_name)
            name, _ = os.path.splitext(filename)
            new_file_path = os.path.join(directory, f"{name}.npy")
            self.model.B = np.load(new_file_path)
        else:
            self.model.load_state_dict(torch.load(file_name))


class DeepFM(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        submission_dataloader,
        seen_items,
        args,
    ):
        super(DeepFM, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            submission_dataloader,
            seen_items,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        rec_data_iter = tqdm.tqdm(
            dataloader,
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        if mode == "train":
            """
            train DataLoader를 사용하여 모델을 한 epoch 동안 학습

            1. 모델을 학습 모드로 설정
            2. 데이터 배치를 반복하여 손실을 계산
            3. 역전파 수행하며, 모델의 매개변수 업데이트
            4. 평균 손실 기록
            """
            self.model.train()

            num_batches = len(dataloader)
            train_loss = 0
            for i, batch in enumerate(rec_data_iter):
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                self.optim.zero_grad()
                output = self.model(X)
                loss = nn.BCELoss()(output, y.float())
                loss.backward()
                self.optim.step()
                train_loss += loss.item()

                if (i + 1) % 1000 == 0 or (i + 1) == num_batches:
                    avg_loss = train_loss / (i + 1)

            avg_loss = train_loss / num_batches
            rec_data_iter.write(f"Average Loss: {avg_loss:.6f}")
            wandb.log({"loss": avg_loss})

        elif mode == "valid":
            """
            valid DataLoader를 사용하여 모델의 성능을 검증

            1. 모델을 평가 모드로 설정
            2. 데이터 배치를 반복하여 예측 결과를 계산
            3. 예측 결과와 실제 레이블을 비교하여 Accuracy를 계산하고 기록

            Returns:
                list[float]: Accuracy(백분율)가 포함된 리스트
            """
            self.model.eval()

            correct_result_sum = 0
            total_samples = 0
            with torch.no_grad():
                for i, batch in enumerate(rec_data_iter):
                    X, y = batch
                    X, y = X.to(self.device), y.to(self.device)

                    output = self.model(X)
                    result = torch.round(output)
                    correct_result_sum += (result == y).sum().float()
                    total_samples += y.size(0)

            acc = correct_result_sum / total_samples * 100
            rec_data_iter.write("Average Acc: {:.2f}%".format(acc.item()))

            return [acc.item()]

        else:
            """
            submission DataLoader를 사용하여 모델의 예측 결과를 생성하고, 사용자별로 추천할 아이템을 추출

            1. 모델을 평가 모드로 설정
            2. 데이터 배치를 반복하여 예측 결과를 계산
            3. 각 사용자에 대해 Top-K 추천 아이템을 추출하고, 실제 아이템과 비교하여 결과 반환

            Returns:
                Union[list[int], None]:
                - mode가 test인 경우, None을 반환하고 Ranking 점수 계산
                - mode가 submission인 경우, 각 사용자에 대한 추천 아이템 리스트를 반환
            """
            self.model.eval()

            outputs, users, items, answers = [], [], [], []
            with torch.no_grad():
                for i, batch in enumerate(rec_data_iter):
                    X, y = batch
                    X, y = X.to(self.device), y.to(self.device)

                    output = self.model(X)
                    outputs.append(output.cpu())
                    users.append(X[:, 0].cpu())
                    items.append(X[:, 1].cpu())
                    answers.append(y[:].cpu())

            # 모든 batch의 출력을 하나의 tensor로 결합
            all_outputs = torch.cat(outputs, dim=0)
            all_users = torch.cat(users, dim=0)
            all_items = torch.cat(items, dim=0)
            all_answers = torch.cat(answers, dim=0)

            user_data = defaultdict(lambda: {"ratings": [], "items": [], "answers": []})
            for user, rating, item, answer in zip(all_users, all_outputs, all_items, all_answers):
                user_data[user.item()]["ratings"].append(rating)
                user_data[user.item()]["items"].append(item)
                user_data[user.item()]["answers"].append(answer)

            predicted, actual = [], []
            for user, data in tqdm.tqdm(user_data.items()):
                user_ratings = torch.tensor(data["ratings"])
                user_items = torch.tensor(data["items"])
                user_answers = torch.tensor(data["answers"])

                # 이미 본 아이템 마스킹
                if user in self.seen_items:
                    seen_items_set = torch.tensor(self.seen_items[user], device=user_ratings.device)
                    mask = torch.isin(user_items, seen_items_set)
                    user_ratings[mask] = -1

                # Top-10 아이템 추출
                _, top_k_indices = torch.topk(user_ratings, 10)
                predicted.append(user_items[top_k_indices].tolist())
                actual_items = user_items[user_answers == 1]
                actual.append(actual_items.tolist())

            if mode == "submission":
                return predicted
            else:
                self.get_full_sort_score(epoch, actual, predicted)


class EASE(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        submission_dataloader,
        seen_items,
        args,
    ):
        super(EASE, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            submission_dataloader,
            seen_items,
            args,
        )

    def iteration(self, _, data, mode="train"):

        if mode in ("train", "test"):
            """
            train user-item interaction matrix를 사용하여 모델을 학습
            - mode가 train인 경우, train 데이터 학습
            - mode가 test인 경우, 전체 데이터에 대해 재학습
            """
            self.model.train(data)

            if mode == "test":
                self.save(self.args.checkpoint_path)
                print(f"Saved Model: {self.args.checkpoint_path}")

        elif mode == "valid":
            """
            train, valid user-item interaction matrix를 사용하여 모델의 성능을 검증

            1. train 데이터에서 평가 항목을 제외
            2. 모델을 사용하여 1번 데이터에 대한 예측 점수를 계산
            3. 2번 데이터와 valid 데이터를 비교해 Ranking 평가지표를 계산하고 기록

            Returns:
                list[float]: RECALL@10이 포함된 리스트
            """
            train_data = self.seen_items
            test_data = data
            top_k = 10

            # EASE 모델로 예측 점수 계산
            predictions = self.model.predict(train_data)

            # 학습 데이터에서 평가 항목 제외
            predictions[train_data.nonzero()] = -np.inf

            # 평가 지표 계산
            n10 = ndcg_binary_at_k_batch(predictions, test_data, top_k)
            r10 = recall_at_k_batch(predictions, test_data, top_k)

            loss = self.model.loss_function_ease(train_data)

            post_fix = {
                "loss": f"{loss:.6f}",
                "RECALL@10": f"{np.nanmean(r10):.4f}",
                "NDCG@10": f"{np.nanmean(n10):.4f}"
            }
            print(post_fix)
            wandb.log(post_fix)

            return [np.nanmean(r10)]

        else:
            """
            submission user-item interaction matrix를 사용하여 모델의 예측 결과를 생성하고, 사용자별로 추천할 아이템을 추출

            Returns:
                list[int]: 각 사용자에 대한 추천 아이템 리스트를 반환
            """
            interaction_matrix = data

            predictions = self.model.predict(interaction_matrix)
            predictions[interaction_matrix.nonzero()] = -np.inf

            top_items_per_user = []
            for user_idx in range(predictions.shape[0]):
                top_items = np.argsort(predictions[user_idx])[-10:][::-1]
                top_items_per_user.append(top_items)

            return top_items_per_user


class EASER(EASE):
    """ EASE 모델을 상속받아 동일한 기능을 제공하는 클래스 """
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        submission_dataloader,
        seen_items,
        args,
    ):
        super(EASER, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            submission_dataloader,
            seen_items,
            args,
        )


class MultiVAE(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        submission_dataloader,
        seen_items,
        args,
    ):
        super(MultiVAE, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            submission_dataloader,
            seen_items,
            args,
        )

    def iteration(self, epoch, data, mode="train"):

        if mode in ("train", "test"):
            """
            train user-item interaction matrix를 사용하여 모델을 학습
            - mode가 train인 경우, train 데이터 학습
            - mode가 test인 경우, 전체 데이터에 대해 재학습

            1. 모델을 학습 모드로 설정
            2. 데이터 배치를 반복하여 손실을 계산
            3. 역전파 수행하며, 모델의 매개변수 업데이트
            4. mode가 train인 경우, 평균 손실 기록
               mode가 test인 경우, 모델 상태 저장
            """
            self.model.train()
            train_data = data

            total_loss = 0
            for start in range(0, train_data.shape[0], self.args.batch_size):
                end = min(start + self.args.batch_size, train_data.shape[0])
                batch = torch.FloatTensor(train_data[start:end].toarray()).to(self.device)

                self.optim.zero_grad()
                recon_batch, mean, logvar = self.model(batch)
                loss = self.model.loss_function_multivae(batch, recon_batch, mean, logvar, self.args.kl_beta)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

            if mode == "train":
                avg_loss = total_loss / train_data.shape[0]
                print(f"[{epoch} Epoch] Average Loss: {avg_loss:.6f}")
                wandb.log({"loss": avg_loss})
            else:
                self.save(self.args.checkpoint_path)
                print(f"Saved Model: {self.args.checkpoint_path}")

        elif mode == "valid":
            """
            train, valid user-item interaction matrix를 사용하여 모델을 학습

            1. 모델을 평가 모드로 설정
            2. train 데이터에서 평가 항목을 제외
            2. 모델을 사용하여 2번 데이터에 대한 예측 점수를 계산
            3. 3번 데이터와 valid 데이터를 비교해 Ranking 평가지표를 계산하고 기록

            Returns:
                list[float]: RECALL@10이 포함된 리스트
            """
            self.model.eval()
            train_data = self.seen_items
            valid_data = data

            # loss와 평가 지표를 담을 리스트 생성
            total_loss_list = []
            n10_list, r10_list = [], []

            with torch.no_grad():
                for start in range(0, train_data.shape[0], self.args.batch_size):
                    end = min(start + self.args.batch_size, train_data.shape[0])
                    batch = torch.FloatTensor(train_data[start:end].toarray()).to(self.device)
                    heldout_batch = valid_data[start:end]

                    recon_batch, mean, logvar = self.model(batch)
                    loss = self.model.loss_function_multivae(recon_batch, batch, mean, logvar, self.args.kl_beta)
                    total_loss_list.append(loss.item())

                    # 평가된 아이템 제외
                    recon_batch = recon_batch.cpu().numpy()
                    batch = batch.cpu().numpy()
                    recon_batch[batch.nonzero()] = -np.inf

                    # NDCG@10, Recall@10 계산
                    n10 = ndcg_binary_at_k_batch(recon_batch, heldout_batch, 10)
                    r10 = recall_at_k_batch(recon_batch, heldout_batch, 10)

                    n10_list.append(n10)
                    r10_list.append(r10)

            n10_list = np.concatenate(n10_list)
            r10_list = np.concatenate(r10_list)

            post_fix = {
                "RECALL@10": "{:.4f}".format(np.nanmean(r10_list)),
                "NDCG@10": "{:.4f}".format(np.nanmean(n10_list)),
            }
            print(post_fix)
            wandb.log(post_fix)

            return [np.nanmean(r10_list)]

        else:
            """
            submission user-item interaction matrix를 사용하여 모델의 예측 결과를 생성하고, 사용자별로 추천할 아이템을 추출

            Returns:
                list[int]: 각 사용자에 대한 추천 아이템 리스트를 반환
            """
            self.model.eval()
            interaction_matrix = data

            with torch.no_grad():
                interaction_matrix_tensor = torch.FloatTensor(interaction_matrix.toarray()).to(self.device)
                predictions, _, _ = self.model(interaction_matrix_tensor)

            predictions[interaction_matrix.nonzero()] = -np.inf
            top_items_per_user = predictions.topk(10, dim=1)[1].cpu().numpy()

            return top_items_per_user
