import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import VisionTransformer
from models.base import BaseModel


class CNNVideoTransformer(BaseModel):
    def __init__(self):
        # TODO: define model here and pass args to parentClass
        super().__init__()
        self.feature_extractor = NotImplemented
        self.transformer = NotImplemented

        self.cls_acc = NotImplemented
        self.frame_acc = NotImplemented

    def forward(self, x):
        """
        :param x: imgs, N x L x C x H x W, L is sequence length
        :return:
        """
        # TODO: Implement forward
        N, L = x.shape[:2]
        x = x.flatten(start_dim=0, end_dim=2)  # NL x C x H x W
        features = self.feature_extractor(x)  # NL x C' x H' x W'
        features = features.view(N, L, *(features.shape[1:]))  # N x L x C' x H' x W'
        frame_features = self.extract_frame_features(features)  # N x L x C'
        # Note that video-level cls and frame lvl cls can be different, don't necessary to form a vector.
        cls_logits, frame_logits = self.transformer(frame_features)  # N x Cls1, N x L x Cls2
        return cls_logits, frame_logits

    def training_step(self, batch, batch_idx):
        x, y_cls, y_frames = batch  # y_cls: N, y_frames: N x L

        cls_logits, frame_logits = self.forward(x)
        loss_cls = F.cross_entropy(cls_logits, y_cls)
        loss_frames = F.cross_entropy(frame_logits, y_frames, reduction='none').sum(1).mean()
        loss = loss_cls + loss_frames

        tensorboard_logs = {'loss': loss, 'loss_cls': loss_cls, 'loss_frames': loss_frames}

        return {'loss': loss, 'logs': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y_cls, y_frame = batch  # y_cls: N, y_frames: N x L

        cls_logits, frame_logits = self.forward(x)
        loss_cls = F.cross_entropy(cls_logits, y_cls)
        loss_frames = F.cross_entropy(frame_logits, y_frame, reduction='none').sum(1).mean()
        loss = loss_cls + loss_frames

        _, y_cls_hat = torch.max(cls_logits, dim=1)
        _, y_frame_hat = torch.max(frame_logits, dim=2)
        cls_acc = self.cls_acc(y_cls_hat, y_cls)
        frame_acc = self.frame_acc(y_frame_hat.flatten(), y_frame.flatten())  # TODO: Verify this

        return {'val_loss': loss, 'val_cls_acc': cls_acc, 'val_frame_acc': frame_acc}

    def validation_epoch_end(self, outputs):


    def test_step(self, batch, batch_idx):
        # TODO: Implement test step
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]
