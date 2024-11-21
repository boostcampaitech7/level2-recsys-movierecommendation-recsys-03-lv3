import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
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
            bar_format="{l_bar}{r_bar}",
        )

        if mode == "train":
            self.model.train()

            for x, y in rec_data_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.optim.zero_grad()
                output = self.model(x)
                loss = nn.BCELoss()(output, y.float())
                loss.backward()
                self.optim.step()         
        else:
            self.model.eval()
        
            pred_list = None
            answer_list = None
            for i, batch in enumerate(rec_data_iter):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Predict scores for all items
                rating_pred = self.model(x)  # DeepFM Forward

                # Exclude already interacted items
                rating_pred = rating_pred.cpu().data.numpy()
                user_ids = x[:, 0].cpu().numpy()  # User IDs from batch
                rating_pred[self.train_matrix[user_ids].toarray() > 0] = 0

                # Get top 10 recommendations
                ind = np.argpartition(rating_pred, -10)[:, -10:]  # Top-10 indices
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[
                    np.arange(len(rating_pred))[:, None], arr_ind_argsort
                ]

                # Append results
                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = y.cpu().numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, y.cpu().numpy(), axis=0)

            recall_5 = recall_at_k(answer_list, pred_list, topk=5)
            recall_10 = recall_at_k(answer_list, pred_list, topk=10)

            post_fix = {
                "RECALL@5": "{:.4f}".format(recall_5),
                "RECALL@10": "{:.4f}".format(recall_10),
            }
            print(post_fix)
            wandb.log(post_fix)

            return [recall_5, recall_10]
