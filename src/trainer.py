import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm
import wandb

from .utils import recall_at_k


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
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
        args,
    ):
        super(DeepFM, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
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
            for x, y in rec_data_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.model.train()
                self.optim.zero_grad()
                output = self.model(x)
                loss = nn.BCELoss()(output, y.float())
                loss.backward()
                self.optim.step()
        elif mode == "submission":
            pass
        else:
            predicted = []
            actual = []
            for x, y in rec_data_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.model.eval()
                output = self.model(x)

                # 예측을 상위 10개로 정렬
                _, top_k_indices = torch.topk(output, k=10, dim=0)

                # 예측 결과를 저장
                predicted.append(top_k_indices.cpu().numpy().tolist())
                actual.append(y.cpu().numpy().tolist())

            recall_5 = recall_at_k(actual, predicted, topk=5)
            recall_10 = recall_at_k(actual, predicted, topk=10)

            post_fix = {
                "RECALL@5": "{:.4f}".format(recall_10),
                "RECALL@10": "{:.4f}".format(recall_5),
            }
            print(post_fix)
            wandb.log(post_fix)
