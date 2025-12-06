import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import RawPointCloudDataset
from criterion import MAECriterion

import yaml
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
    ):
        with open("training/train_cfg.yaml", "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.epoch = spec["epoch"]
        self.model = model
        self.opt = optim.Adam(self.model.parameters(), lr=spec["learning_rate"])
        self.criterion = MAECriterion()

        self.stamp_every = spec["stamp_every"]
        
        self.train_ds = RawPointCloudDataset(
            min_pointclouds=spec["min_pointclouds"],
            max_pointclouds=spec["max_pointclouds"],
            max_seq_length=spec["max_seq_length"],
            input_seq_length=spec["input_seq_length"],
        )
        self.train_dl = DataLoader(
            self.train_ds,
            collate_fn=self.train_ds.collater,
            batch_size=spec["batch_size"]
        )

    def train(self):
        improve = False
        best_val_loss = 1e10
        train_loss_history = []

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        for epoch in range(self.epoch):
            self.model.train()
            total_training_loss = 0
            for step, sample in enumerate(self.train_dl):
                results = self.criterion(self.model, sample)

                train_loss = results["loss"]
                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()
                torch.cuda.empty_cache()

                total_training_loss += results["loss"].item()

                if (step + 1) % self.stamp_every == 0:
                    train_loss_history.append(total_training_loss / (step + 1))
                    print(f"Epoch: {epoch + 1} / {self.epoch}\t Step: {step + 1} / {len(self.train_dl)}\
\tTrain loss: {total_training_loss / (step + 1):.4f}")
                    
        plt.plot(train_loss_history)
        plt.show()

