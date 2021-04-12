import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .base import BaseModel
from models import VideoPerformer


class PerformerTrainer(BaseModel):
    def __init__(self, model_kwargs, trainer_kwargs):
        super().__init__(**trainer_kwargs)
        self.model = VideoPerformer(**model_kwargs)

        self.cls_acc = pl.metrics.Accuracy()
        self.frame_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_frame, y_cls = batch  # y_cls: N, y_frames: N x L

        cls_logits, frame_logits = self.forward(x)
        loss_cls = F.cross_entropy(cls_logits, y_cls)
        loss_frames = F.cross_entropy(frame_logits.flatten(0,1), y_frame.flatten())
        loss = loss_cls + loss_frames

        tensorboard_logs = {'loss': loss, }
        tqdm_logs = {'loss_cls': loss_cls, 'loss_frames': loss_frames}

        return {'loss': loss, 'progress_bar': tqdm_logs, 'logs': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y_frame, y_cls = batch  # y_cls: N, y_frames: N x L

        cls_logits, frame_logits = self.forward(x)
        loss_cls = F.cross_entropy(cls_logits, y_cls)
        loss_frames = F.cross_entropy(frame_logits.flatten(0,1), y_frame.flatten())
        loss = loss_cls + loss_frames

        _, y_cls_hat = torch.max(cls_logits, dim=-1)
        _, y_frame_hat = torch.max(frame_logits, dim=-1)
        cls_acc = self.cls_acc(y_cls_hat, y_cls)
        frame_acc = self.frame_acc(y_frame_hat.flatten(), y_frame.flatten())  # TODO: Verify this

        return {'val_loss': loss, 'val_cls_acc': cls_acc, 'val_frame_acc': frame_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_cls_acc = torch.stack([x['val_cls_acc'] for x in outputs]).mean()
        avg_frame_acc = torch.stack([x['val_frame_acc'] for x in outputs]).mean()
        logs = {'avg_val_loss': avg_loss, 'avg_val_cls_acc': avg_cls_acc, 'avg_val_frame_cls': avg_frame_acc}

        self.cls_acc.reset()
        self.frame_acc.reset()
        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):

        names = ['pos_embed', 'cls_token']
        named_params = dict(self.model.named_parameters())
        no_decay_params = [named_params[name] for name in names]
        default_params = [v for k, v in named_params.items() if k not in names]
        # default_params = list(self.model.cls_head.parameters()) + list(self.model.token_head.parameters())
        optimizer = torch.optim.Adam([
            {'params': default_params},
            {'params': no_decay_params, 'weight_decay': 0.}
        ],
            lr=self.lr, weight_decay=0
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]