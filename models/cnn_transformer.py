# Implement per-token output transformer
# Code is based on
#     https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from functools import partial
from timm.models.vision_transformer import VisionTransformer, trunc_normal_
from timm.models.vision_transformer import vit_small_patch16_224, vit_small_resnet26d_224, vit_base_patch16_224, \
    vit_base_patch16_384, vit_base_patch32_384, vit_large_patch16_224, vit_large_patch16_384, vit_large_patch32_384, \
    vit_huge_patch16_224, vit_huge_patch32_384, vit_base_resnet26d_224, vit_base_resnet50d_224, vit_small_resnet50d_s3_224


class VideoCNNTransformer(VisionTransformer):
    def __init__(self, video_length, num_token_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_length = video_length
        self.num_token_classes = num_token_classes

        token_length = video_length
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1+token_length, self.embed_dim))

        # Classifier head
        self.cls_head = nn.Linear(
            self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.token_head = nn.Linear(
            self.num_features, num_token_classes) if num_token_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.cnn = models.resnet18(True)
        self.cnn2 = nn.Linear(1000, self.num_features)

    def get_classifier(self):
        return self.cls_head, self.token_head

    def reset_classifier(self, num_classes, num_token_classes, global_pool=''):
        self.num_classes = num_classes
        self.num_token_classes = num_token_classes
        assert num_classes > 0
        self.cls_head = nn.Linear(self.embed_dim, num_classes)
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.zeros_(self.cls_head.bias)
        assert num_token_classes > 0
        self.token_head = nn.Linear(self.embed_dim, num_token_classes)
        nn.init.normal_(self.token_head.weight, std=0.01)
        nn.init.zeros_(self.token_head.bias)

    def forward_features(self, x):
        """
        :param x: B x L x C x H x W
        :return: x: B x (1+Lhw) x C'
        """
        B, L = x.shape[:2]

        x = x.flatten(start_dim=0, end_dim=1)
        x = self.cnn2(self.cnn(x))
        x = x.reshape(B, L, x.shape[-1]
                      )

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)  # B x (1+LHW) x C
        cls_logits = self.cls_head(x[:, 0])  # B x C'
        token_logits = self.token_head(x[:, 1:])  # N x L x C''
        return cls_logits, token_logits


def video_cnnt_small_patch16_224(seq_len, num_classes, num_token_classes):
    model = VideoCNNTransformer(seq_len, num_token_classes,
                                patch_size=16, embed_dim=768, depth=8, num_heads=8,
                                mlp_ratio=3., qk_scale=768 ** -0.5)

    model.reset_classifier(num_classes=num_classes,
                           num_token_classes=num_token_classes)
    return model


def video_cnnt_base_patch_224(seq_len, num_classes, num_token_classes):
    model = VideoCNNTransformer(seq_len, num_token_classes, fuse_patch,
                                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6))

    model.reset_classifier(num_classes=num_classes,
                           num_token_classes=num_token_classes)
    return model


def video_cnnt_large_patch_224(seq_len, num_classes, num_token_classes):
    model = VideoCNNTransformer(seq_len, num_token_classes, fuse_patch,
                                patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6))

    model.reset_classifier(num_classes=num_classes,
                           num_token_classes=num_token_classes)
    return model


if __name__ == '__main__':
    device = 'cuda'
    x = torch.randn(2, 4, 3, 224, 224).to(device)
    model = VideoCNNTransformer(4, 8,
                                img_size=224, num_classes=6).to(device)
    cls_logits, token_logits = model(x)
    print(cls_logits.shape, token_logits.shape)
