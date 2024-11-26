import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm
import wandb

from .utils import recall_at_k, precision_at_k, mapk, ndcg_k, random_neg


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        submission_dataloader,
        seen_items,
        args,
    ):

        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        if self.device == "cuda":
            self.cuda_condition = torch.cuda.is_available()
            if self.cuda_condition:
                self.model.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.submission_dataloader = submission_dataloader
        self.seen_items = seen_items

        self.optim = Adam(
            self.model.parameters(),
            lr=0.001,
        )

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, actual, predicted):
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
        print("Epoch: ", epoch)
        print(post_fix)
        wandb.log(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
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
            rec_data_iter.write(f"Final Average Loss: {avg_loss:.6f}")
            wandb.log({"loss": avg_loss})

        elif mode == "valid":
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
            rec_data_iter.write("Final Acc : {:.2f}%".format(acc.item()))

            return [acc.item()]

        else:
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

            unique_users = all_users.unique()
            predicted, actual = [], []
            for user in tqdm.tqdm(unique_users):
                # 현재 사용자에 대한 평점과 아이템 ID 필터링
                user_mask = (all_users == user)
                user_ratings = all_outputs[user_mask]
                user_items = all_items[user_mask]
                user_answers = all_answers[user_mask]

                # 이미 본 아이템 마스킹
                if user.item() in self.seen_items:
                    for idx, item in enumerate(user_items):
                        if item.item() in self.seen_items[user.item()]:
                            user_ratings[idx] = -1

                # Top-10 아이템 추출
                _, top_k_indices = torch.topk(user_ratings, 10)
                predicted.append(user_items[top_k_indices].tolist())
                actual_items = user_items[user_answers == 1]
                actual.append(actual_items.tolist())

            if mode == "submission":
                return predicted
            else:
                self.get_full_sort_score(epoch, actual, predicted)


class BERT4Rec(Trainer):
    def __init__(self,
        model,
        train_dataloader,
        eval_dataloader,
        submission_dataloader,
        args,
        user_train = None,
        user_valid = None,
    ):
        super(BERT4Rec, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            submission_dataloader,
            None,
            args,
        )
        self.user_train = user_train
        self.user_valid = user_valid
        self.num_user = args.model_args.BERT4Rec.num_user
        self.num_item = args.model_args.BERT4Rec.num_item
        
    def iteration(self, epoch, dataloader, mode="train"):
        if mode == "train":
            self.model.train()
            self.model.to(self.device)
            
            tqdm_bar = tqdm.tqdm(dataloader)
            train_loss = 0
            num_batches = len(dataloader)
            for i, (log_seqs, labels) in enumerate(tqdm_bar):
                logits = self.model(log_seqs)

                # size matching
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1).to(self.device)

                self.optim.zero_grad()
                loss = nn.CrossEntropyLoss(ignore_index=0)(logits, labels)
                loss.backward()
                self.optim.step()

                tqdm_bar.set_description(f'Epoch: {epoch:3d}| Step: {i:3d}| Train loss: {loss:.5f}')
                train_loss += loss.item()

            avg_loss = train_loss / num_batches
            wandb.log({"loss": avg_loss})

        elif mode == "valid":
            self.model.eval()

            # 메트릭 초기화
            NDCG = 0.0
            HIT = 0.0
            total_users = 0

            with torch.no_grad():
                for log_seqs, labels in tqdm.tqdm(dataloader, desc=f"{mode.upper()} Epoch {epoch}"):
                    # 배치 데이터 로드
                    print("log_seqs:", log_seqs)
                    print("labels:", labels)
                    log_seqs = log_seqs.to(self.device)
                    labels = labels.to(self.device)

                    # 예측 수행
                    predictions = self.model(log_seqs)
                    predictions = predictions.cpu().numpy()
                    print(predictions.shape)

                    # 각 사용자별로 상위 K개 추천
                    for user_idx, true_items in enumerate(labels):
                        print("true_items:", true_items)
                        rated_items = set(true_items.cpu().numpy())
                        print("before:", rated_items)
                        rated_items.discard(0)  # 패딩 토큰 제외
                        print("after:", rated_items)

                        # 이미 본 아이템 제외
                        predictions[user_idx][list(rated_items)] = float("inf")

                        # Top-10 추천
                        top_k_items = np.argsort(predictions[user_idx])[:10]
                        actual_items = set(true_items.cpu().numpy())

                        # 메트릭 계산
                        if len(actual_items) > 0:
                            if len(set(top_k_items) & actual_items) > 0:
                                HIT += 1
                            rank = next((i for i, item in enumerate(top_k_items) if item in actual_items), None)
                            if rank is not None:
                                NDCG += 1 / np.log2(rank + 2)
                        total_users += 1

            # 평균 메트릭 출력
            NDCG /= total_users
            HIT /= total_users
            print(f"NDCG@10: {NDCG:.4f} | HIT@10: {HIT:.4f}")
            return [NDCG, HIT]

        else:
            self.model.eval()

            results = {}
            with torch.no_grad():
                for user in tqdm.tqdm(range(self.num_user), desc=f"{mode.upper()} Epoch {epoch}"):
                    seq = (self.user_train[user] + [self.num_item + 1])[-self.args.model_args.BERT4Rec.max_len:]
                    rated = set(self.user_train[user])  # 이미 본 아이템들

                    # 모든 아이템에 대해 점수 계산
                    seq_tensor = torch.LongTensor([seq]).cpu()
                    predictions = -self.model(seq_tensor).cpu().numpy()  # 점수는 음수로 변환해 낮은 값이 상위로 오도록

                    # 이미 본 아이템 제외
                    valid_rated = [i for i in rated if i < predictions[0].shape[0]]
                    predictions[0][list(valid_rated)] = float("inf")  # 이미 본 아이템의 점수는 무한대로 설정

                    # 상위 K개의 아이템 추출
                    top_k_items = np.argsort(predictions[0])[:10]  # Top-10 아이템
                    results[user] = top_k_items.tolist()

            if mode == "submission": return results
            else:
                # 평균 메트릭 출력
                NDCG /= total_users
                HIT /= total_users
                print(f"NDCG@10: {NDCG:.4f} | HIT@10: {HIT:.4f}")
                return [NDCG, HIT]