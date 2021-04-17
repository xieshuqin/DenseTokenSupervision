import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from functools import partial

from models.transformer import video_vit_small_patch16_224, video_vit_base_patch_224
from models.cnn_transformer import video_cnnt_small_patch16_224
from trainer import Trainer

model_factory = {
    'video_vit_small': partial(video_vit_small_patch16_224, num_classes=4, num_token_classes=288, fuse_patch=True),
    'video_vit_base': partial(video_vit_base_patch_224, num_classes=4, num_token_classes=288, fuse_patch=True),
    'video_cnnt_small': partial(video_cnnt_small_patch16_224, num_classes=4, num_token_classes=288),
}

dataset_kwargs = {
    'finegym': partial(dict, frame_size=224),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(model_factory.keys()), type=str)
    parser.add_argument('--dataset', default='finegym',
                        choices=list(dataset_kwargs.keys()), type=str)
    parser.add_argument('--seq_len', default=8, help='Video sequence length')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--bs', default=16, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--w_video', default=1., type=float,
                        help='weight for video classification loss')
    parser.add_argument('--w_frame', default=1., type=float,
                        help='weight for frame classification loss')
    return parser.parse_args()


def main():
    args = parse_args()
    model = model_factory[args.model](seq_len=args.seq_len)
    model = Trainer(model, args.w_video, args.w_frame,
                    dataset_type=args.dataset, batch_size=args.bs, num_epochs=args.epochs, lr=args.lr,
                    **dataset_kwargs[args.dataset](num_frames_per_video=args.seq_len))
    # load checkpoint
    # model = model_factory[args.task]
    # model = model.load_from_checkpoint('./lightning_logs/version_22/checkpoints/epoch=249-step=109499.ckpt')
    trainer = pl.Trainer(
        max_epochs=args.epochs, gpus=1, default_root_dir='./runs/%s' % args.model,
        gradient_clip_val=1., log_every_n_steps=1, val_check_interval=100
    )
    trainer.fit(model)
    # trainer.test(model)


if __name__ == '__main__':
    main()
