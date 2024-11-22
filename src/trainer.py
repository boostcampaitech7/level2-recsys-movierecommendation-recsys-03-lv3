import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import tqdm
import wandb

from collections import defaultdict
from .utils import recall_at_k


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        train_matrix,
        args,
    ):

        self.args = args
        self.model = model
        self.device = args.device
        if self.device == "cuda":
            self.cuda_condition = torch.cuda.is_available()
            if self.cuda_condition:
                self.model.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader
        self.train_matrix = train_matrix

        self.optim = Adam(
            self.model.parameters(),
            lr=0.01,
        )

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

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
        test_dataloader,
        submission_dataloader,
        train_matrix,
        args,
    ):
        super(DeepFM, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            train_matrix,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        rec_data_iter = tqdm.tqdm(
            dataloader,
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}",
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
            K = 10
            for user in unique_users:
                # 현재 사용자에 대한 평점과 아이템 ID 필터링
                user_mask = (all_users == user)
                user_ratings = all_outputs[user_mask]
                user_items = all_items[user_mask]
                user_answers = all_answers[user_mask]

                # Top-K 아이템 추출
                _, top_k_indices = torch.topk(user_ratings, K)
                predicted.append(top_k_indices)
                actual.append(answers)

                top_k_items = user_items[top_k_indices]
                if user == 4058:
                    print(f"User {user.item()}:")
                    for item_id, answer in zip(user_items.numpy(), user_answers.numpy()):
                        print(f"  Item {item_id}: Rating {answer:.4f}")
                    for item_id in zip(top_k_items.numpy()):
                        print(f"  Item {item_id}")



            