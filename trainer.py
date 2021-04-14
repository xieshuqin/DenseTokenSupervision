import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from datasets import build_dataloader


class Trainer(pl.LightningModule):
    def __init__(self, model, w_video, w_frame, dataset_type, batch_size, num_epochs, lr, **dataset_kwargs):
        super().__init__()
        self.model = model
        self.w_video = w_video
        self.w_frame = w_frame

        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.dataset_kwargs = dataset_kwargs

        self.num_epochs = num_epochs
        self.lr = lr

        # dim 0 label, dim 1 pred
        self.video_conf_matrix = torchmetrics.ConfusionMatrix(num_classes=4, compute_on_step=False)
        self.frame_conf_matrix = torchmetrics.ConfusionMatrix(num_classes=288, compute_on_step=False)

    def train_dataloader(self):
        return build_dataloader(self.dataset_type, split='train', batch_size=self.batch_size, **self.dataset_kwargs)

    def val_dataloader(self):
        return build_dataloader(self.dataset_type, split='val', batch_size=self.batch_size, **self.dataset_kwargs)

    def test_dataloader(self):
        return build_dataloader(self.dataset_type, split='test', batch_size=self.batch_size, **self.dataset_kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_frame, y_video = batch  # y_video: N, y_frames: N x L

        video_logits, frame_logits = self.forward(x)
        loss_video = F.cross_entropy(video_logits, y_video)
        loss_frame = F.cross_entropy(frame_logits.flatten(0,1), y_frame.flatten())
        loss = self.w_video * loss_video + self.w_frame * loss_frame

        logs = {'loss': loss, 'loss_video': loss_video, 'loss_frame': loss_frame}
        tqdm_logs = {'loss_video': loss_video, 'loss_frame': loss_frame}

        return {'loss': loss, 'progress_bar': tqdm_logs, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y_frame, y_video = batch  # y_video: N, y_frames: N x L

        video_logits, frame_logits = self.forward(x)
        loss_video = F.cross_entropy(video_logits, y_video)
        loss_frame = F.cross_entropy(frame_logits.flatten(0,1), y_frame.flatten())
        loss = self.w_video * loss_video + self.w_frame * loss_frame

        self.video_conf_matrix.update(video_logits.softmax(dim=1), y_video)
        self.frame_conf_matrix.update(frame_logits.flatten(0,1).softmax(dim=1), y_frame.flatten())

        return {'val_loss': loss, 'val_loss_video': loss_video, 'val_loss_frame': loss_frame}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss_video = torch.stack([x['val_loss_video'] for x in outputs]).mean()
        avg_loss_frame = torch.stack([x['val_loss_frame'] for x in outputs]).mean()

        video_matrix = self.video_conf_matrix.compute()
        video_top1_acc = video_matrix.diagonal().sum() / (video_matrix.sum())
        video_mean_acc = (video_matrix.diagonal() / (video_matrix.sum(dim=0))).mean()

        frame_matrix = self.frame_conf_matrix.compute()
        frame_top1_acc = frame_matrix.diagonal().sum() / (frame_matrix.sum())
        frame_mean_acc = (frame_matrix.diagonal() / (frame_matrix.sum(dim=0))).mean()

        logs = {'val_loss': avg_loss, 'val_loss_video': avg_loss_video, 'val_loss_frame': avg_loss_frame,
                'video_top1_acc': video_top1_acc, 'video_mean_acc': video_mean_acc,
                'frame_top1_acc': frame_top1_acc, 'frame_mean_acc': frame_mean_acc}

        self.video_conf_matrix.reset()
        self.frame_conf_matrix.reset()
        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]