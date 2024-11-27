import torch
import torch.nn as nn
import tqdm
import wandb
import numpy as np
from torch.optim import Adam

from .utils import recall_at_k, precision_at_k, mapk, ndcg_k


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        submission_dataloader,
        user_groups,
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
        self.submission_dataloader = submission_dataloader
        self.seen_items = user_groups

        self.optim = Adam(
            self.model.parameters(),
            lr=0.01,
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
        user_groups,
        args,
    ):
        super(DeepFM, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            submission_dataloader,
            user_groups,
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

            actual, predicted = [], []
            for u, u_items, u_reviews, u_seen_items in tqdm.tqdm(self.user_groups):
                u_items = np.array(u_items)
                u_reviews = np.array(u_reviews)

                user_col = torch.full((len(u_items),), u, device=self.device)
                item_col = torch.tensor(u_items, device=self.device)

                X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1)], dim=1)
                y = torch.tensor(u_reviews)

                output = self.model(X)
                output = output.cpu().detach().numpy()
                
                if u == 1: print("u_items:", u_items)
                if u == 1: print("u_seen_items:", u_seen_items)
                if u == 1: print("mask 전 output:", output)
                mask = np.isin(u_items, u_seen_items)
                output[mask] = -1
                if u == 1: print("mask 후 output:", output)

                top_k_idxs = np.argsort(output)[-10:][::-1]
                predicted.append(u_items[top_k_idxs].tolist())
                if u == 1: print(predicted)

            if mode == "submission":
                return predicted
            else:
                indices = np.where(u_reviews == 1)
                actual_items = u_items[indices]
                actual.append([item for item in actual_items if item not in u_seen_items])
                self.get_full_sort_score(epoch, actual, predicted)
