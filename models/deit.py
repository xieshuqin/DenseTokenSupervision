# Code adapted from https://github.com/facebookresearch/deit/blob/main/models.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
from timm.models.vision_transformer import VisionTransformer, trunc_normal_
from timm.models.vision_transformer import vit_small_patch16_224, vit_small_resnet26d_224, vit_base_patch16_224, \
    vit_base_patch16_384, vit_base_patch32_384, vit_large_patch16_224, vit_large_patch16_384, vit_large_patch32_384, \
    vit_huge_patch16_224, vit_huge_patch32_384, vit_base_resnet26d_224, vit_base_resnet50d_224, \
    vit_small_resnet50d_s3_224


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return x, x_dist
        # if self.training:
        #     return x, x_dist
        # else:
        #     # during inference, return the average of both classifier predictions
        #     return (x + x_dist) / 2

    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.head.weight, std=0.01)
            nn.init.zeros_(self.head.bias)
            self.head_dist = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.head_dist.weight, std=0.01)
            nn.init.zeros_(self.head_dist.bias)
        else:
            self.head = nn.Identity()
            self.head_dist = nn.Identity()


def load_pretrained(model, pretrained_model):
    pretrained_state_dict = pretrained_model.state_dict()
    pretrained_pos_embed = pretrained_state_dict['pos_embed']
    del pretrained_state_dict['pos_embed']

    model.load_state_dict(pretrained_state_dict, strict=False)
    model.pos_embed.data[0, :2] = pretrained_pos_embed.data[0, [0]].repeat(2, 1)
    model.pos_embed.data[0, 2:] = pretrained_pos_embed.data[0, 1:]
    model.dist_token.data = pretrained_state_dict['cls_token'].data


def deit_small_patch16_224(num_classes, pretrained=True):
    model = DistilledVisionTransformer(
                                   patch_size=16, embed_dim=768, depth=8, num_heads=8,
                                   mlp_ratio=3., qk_scale=768 ** -0.5)
    if pretrained:
        pretrained_model = vit_small_patch16_224(pretrained=True)
        load_pretrained(model, pretrained_model)

    model.reset_classifier(num_classes=num_classes)
    return model


def deit_base_patch16_224(num_classes, pretrained=True):
    model = DistilledVisionTransformer(
                             patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained:
        pretrained_model = vit_base_patch16_224(pretrained=True)
        load_pretrained(model, pretrained_model)

    model.reset_classifier(num_classes=num_classes)
    return model


def deit_large_patch16_224(num_classes, pretrained=True):
    model = DistilledVisionTransformer(
                             patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained:
        pretrained_model = vit_large_patch16_224(pretrained=True)
        load_pretrained(model, pretrained_model)

    model.reset_classifier(num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = deit_small_patch16_224(1000, pretrained=True).cuda()
    x = torch.randn(2, 3, 224, 224, device='cuda')
    cls_logits, dist_logit = model(x)

    from torchvision.models.resnet import resnet50
    teacher = resnet50(pretrained=True).cuda()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    dist_target = teacher(x)

    dist_logit = dist_logit.log_softmax(dim=-1)
    dist_target = dist_target.log_softmax(dim=-1)
    # loss_patch = F.kl_div(dist_logit, dist_target, reduction='batchmean', log_target=True)
    loss_patch = F.nll_loss(dist_logit, dist_target.argmax(dim=1))
    loss_patch.backward()
