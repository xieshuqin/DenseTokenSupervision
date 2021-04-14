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


class FusionLayer(nn.Module):
    def __init__(self, num_features, num_classes, patch_length):
        super().__init__()
        # TODO: Replace Conv1d with Linear
        self.patch_length = patch_length
        h = w = int(math.sqrt(self.patch_length))
        self.unflatten = nn.Unflatten(1, (h, w))
        self.head = nn.Conv2d(num_features, num_classes, 2, 2)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (N, hw, C)
        x = self.unflatten(x).permute(0, 3, 1, 2)  # (N, C, h, w)
        x = self.head(x).flatten(2).transpose(1, 2)  # (N, h'w', K)
        return x


class PatchVisionTransformer(VisionTransformer):
    def __init__(self, num_token_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_token_classes = num_token_classes
        self.token_head = FusionLayer(self.embed_dim, self.num_token_classes, self.patch_embed.num_patches)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        cls_logits = self.head(x[:, 0])
        token_logits = self.token_head(x[:, 1:])
        return cls_logits, token_logits

    def get_classifier(self):
        return self.head, self.token_head

    def reset_classifier(self, num_classes, num_token_classes, global_pool=''):
        self.num_classes = num_classes
        self.num_token_classes = num_token_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.head.weight, std=0.01)
            nn.init.zeros_(self.head.bias)
        else:
            self.head = nn.Identity()

        self.token_head = nn.Identity() if num_token_classes <= 0 else \
            FusionLayer(self.num_features, num_token_classes, self.patch_embed.num_patches)


def load_pretrained(model, pretrained_model):
    pretrained_state_dict = pretrained_model.state_dict()
    model.load_state_dict(pretrained_state_dict, strict=False)


def patch_vit_small_patch16_224(num_classes, num_token_classes, pretrained=True):
    model = PatchVisionTransformer(num_token_classes,
                                   patch_size=16, embed_dim=768, depth=8, num_heads=8,
                                   mlp_ratio=3., qk_scale=768 ** -0.5)
    if pretrained:
        pretrained_model = vit_small_patch16_224(pretrained=True)
        load_pretrained(model, pretrained_model)

    model.reset_classifier(num_classes=num_classes, num_token_classes=num_token_classes)
    return model


def patch_vit_base_patch16_224(num_classes, num_token_classes, pretrained=True):
    model = PatchVisionTransformer(num_token_classes,
                             patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained:
        pretrained_model = vit_base_patch16_224(pretrained=True)
        load_pretrained(model, pretrained_model)

    model.reset_classifier(num_classes=num_classes, num_token_classes=num_token_classes)
    return model


def patch_vit_large_patch16_224(num_classes, num_token_classes, pretrained=True):
    model = PatchVisionTransformer(num_token_classes,
                             patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained:
        pretrained_model = vit_large_patch16_224(pretrained=True)
        load_pretrained(model, pretrained_model)

    model.reset_classifier(num_classes=num_classes, num_token_classes=num_token_classes)
    return model


if __name__ == '__main__':
    model = patch_vit_small_patch16_224(1000, 1000, pretrained=True).cuda()
    x = torch.randn(4, 3, 224, 224, device='cuda')
    cls_logits, patch_logit = model(x)

    from models.resnet import resnet50
    teacher = resnet50(pretrained=True).cuda()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    patch_target = teacher(x)

    patch_logit = patch_logit.log_softmax(dim=-1)
    patch_target = patch_target.log_softmax(dim=-1)
    loss_patch = F.kl_div(patch_logit.flatten(0,1), patch_target.flatten(0,1), reduction='batchmean', log_target=True)
    loss_patch.backward()
