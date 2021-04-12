import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from abc import ABCMeta, abstractmethod
from datasets import build_dataloader


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(self, dataset_type, batch_size, num_epochs, lr, **dataset_args):
        super().__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.dataset_args = dataset_args

        self.num_epochs = num_epochs
        self.lr = lr

        self.cls_acc = pl.metrics.Accuracy()
        self.frame_acc = pl.metrics.Accuracy()

    def train_dataloader(self):
        return build_dataloader(self.dataset_type, split='train', batch_size=self.batch_size, **self.dataset_args)

    def val_dataloader(self):
        return build_dataloader(self.dataset_type, split='val', batch_size=self.batch_size, **self.dataset_args)

    def test_dataloader(self):
        return build_dataloader(self.dataset_type, split='test', batch_size=self.batch_size, **self.dataset_args)

