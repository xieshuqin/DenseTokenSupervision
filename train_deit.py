import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from torchmetrics import Accuracy

from torchvision.models import resnet50
from models.deit import deit_small_patch16_224, deit_base_patch16_224, deit_large_patch16_224

from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_cls', default=1., type=float, help='Classification loss weight')
    parser.add_argument('--w_dist', default=1., type=float, help='Distillation loss weight')
    parser.add_argument('--max_epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hard_dist', action='store_true', help='Hard distillation')
    return parser.parse_args()


def trainer():
    args = parse_args()
    bs = args.bs
    lr = args.lr
    max_epochs = args.max_epochs
    w_cls = args.w_cls
    w_dist = args.w_dist
    hard_dist = args.hard_dist
    print_freq = 100
    eval_freq = 1
    save_freq = 1
    exp_name = f'deit/w_cls_{w_cls:.2f}_w_dist_{w_dist:.2f}_hard_dist_{hard_dist}'

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageNet('/home/shuqin/hdd/datasets/imagenet', 'train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=8, shuffle=True, pin_memory=True)

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImageNet('/home/shuqin/hdd/datasets/imagenet', 'val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=4, shuffle=False, pin_memory=True)

    transformer = deit_small_patch16_224(1000, pretrained=True).cuda()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

    resnet = resnet50(pretrained=True).cuda()
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False

    step = 0
    best_acc = 0.
    writter = SummaryWriter(f'experiments/{exp_name}')
    top1_acc, top5_acc = Accuracy(top_k=1, compute_on_step=False).cuda(), Accuracy(top_k=5, compute_on_step=False).cuda()
    for epoch in range(max_epochs):
        transformer.train()
        tbar = tqdm(train_loader)
        for (x, y) in tbar:
            x = x.cuda()
            y = y.cuda()
            cls_logit, dist_logit = transformer(x)
            dist_logit = dist_logit.log_softmax(dim=-1)
            dist_target = resnet(x).log_softmax(dim=-1)
            loss_cls = F.cross_entropy(cls_logit, y)
            if hard_dist:
                loss_dist = F.nll_loss(dist_logit, dist_target.argmax(dim=1))
            else:
                loss_dist = F.kl_div(dist_logit, dist_target, reduction='batchmean', log_target=True)
            loss = w_cls * loss_cls + w_dist * loss_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % print_freq == 0:
                print(f'epoch {epoch} loss {loss.item():.3f}, loss_cls {loss_cls.item():.3f}, loss_dist {loss_dist.item():.3f}')
                writter.add_scalar('Train/Loss', loss.item(), step)
                writter.add_scalar('Train/Loss_cls', loss_cls.item(), step)
                writter.add_scalar('Train/Loss_dist', loss_dist.item(), step)
            step += 1
        scheduler.step()

        if epoch % eval_freq == 0:
            transformer.eval()
            val_loss, val_loss_cls, val_loss_dist = 0., 0., 0.
            num_val = 0.
            with torch.no_grad():
                tbar = tqdm(val_loader)
                for (x, y) in tbar:
                    x = x.cuda()
                    y = y.cuda()
                    cls_logit, dist_logit = transformer(x)
                    dist_logit = dist_logit.log_softmax(dim=-1)
                    dist_target = resnet(x).log_softmax(dim=-1)
                    loss_cls = F.cross_entropy(cls_logit, y)
                    if hard_dist:
                        loss_dist = F.nll_loss(dist_logit, dist_target.argmax(dim=1))
                    else:
                        loss_dist = F.kl_div(dist_logit, dist_target, reduction='batchmean', log_target=True)
                    loss = w_cls * loss_cls + w_dist * loss_dist

                    val_loss += loss.item()
                    val_loss_cls += loss_cls.item()
                    val_loss_dist += loss_dist.item()
                    num_val += 1

                    cls_logit = cls_logit.softmax(dim=-1)
                    top1_acc.update(cls_logit, y)
                    top5_acc.update(cls_logit, y)
            print(f'Val epoch {epoch + 1}, top1: {top1_acc.compute().item():.3f}, top5: {top5_acc.compute().item():.3f}, loss {val_loss / num_val:.4f}, loss_cls {val_loss_cls / num_val:.4f}, loss_dist {val_loss_dist / num_val:.4f}')
            writter.add_scalar('Val/Loss', val_loss/num_val, epoch+1)
            writter.add_scalar('Val/Loss_cls', val_loss_cls/num_val, epoch+1)
            writter.add_scalar('Val/Loss_dist', val_loss_dist/num_val, epoch+1)
            writter.add_scalar('Val/top1_acc', top1_acc.compute().item(), epoch+1)
            writter.add_scalar('Val/top5_acc', top5_acc.compute().item(), epoch+1)

            cur_acc = top1_acc.compute().item()
            if cur_acc > best_acc:
                best_acc = cur_acc
                os.makedirs(f'ckpts/{exp_name}', exist_ok=True)
                obj = {'epoch': epoch,
                       'state_dict': transformer.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'scheduler_state_dict': scheduler.state_dict()}
                torch.save(obj, f'ckpts/{exp_name}/model_best.pth.tar')
            if epoch % save_freq == 0:
                os.makedirs(f'ckpts/{exp_name}', exist_ok=True)
                obj = {'epoch': epoch,
                       'state_dict': transformer.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'scheduler_state_dict': scheduler.state_dict()}
                torch.save(obj, f'ckpts/{exp_name}/model_{epoch}.pth.tar')

            top1_acc.reset()
            top5_acc.reset()
            transformer.train()


if __name__ == '__main__':
    trainer()