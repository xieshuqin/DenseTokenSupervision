# Implement per-token output transformer
# Code is based on
#     https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
from timm.models.vision_transformer import VisionTransformer, trunc_normal_
from timm.models.vision_transformer import vit_small_patch16_224, vit_small_resnet26d_224, vit_base_patch16_224, \
    vit_base_patch16_384, vit_base_patch32_384, vit_large_patch16_224, vit_large_patch16_384, vit_large_patch32_384, \
    vit_huge_patch16_224, vit_huge_patch32_384, vit_base_resnet26d_224, vit_base_resnet50d_224, vit_small_resnet50d_s3_224

class FusionLayer(nn.Module):
    def __init__(self, num_features, num_classes, sequence_length):
        super().__init__()
        # TODO: Replace Conv1d with Linear
        self.sequence_length = sequence_length
        self.layer = nn.Linear(num_features, 1)
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # x: N x LHW x C
        N, C = x.shape[0], x.shape[-1]
        L = self.sequence_length
        attention = torch.softmax(
            self.layer(x).squeeze(-1).reshape(N, L, -1), dim=2)  # N x L x HW

        x = x.reshape(N, L, -1, C)  # N x L x HW x C
        x = (x * attention.unsqueeze(-1)).sum(-2)  # N x L x C
        x = self.head(x)  # N x L x C'
        return x


class VideoTransformer(VisionTransformer):
    def __init__(self, video_length, num_token_classes, fuse_patch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_length = video_length
        self.num_token_classes = num_token_classes
        self.fuse_patch = fuse_patch

        num_patches_per_frame, embed_dim = self.patch_embed.num_patches, self.embed_dim
        token_length = video_length * num_patches_per_frame
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+token_length, embed_dim))

        #  Representation layer
        # if hasattr(self, 'num_features'):
        #     representation_size = self.num_features
        #     self.pre_logits = nn.Sequential(OrderedDict([
        #         ('conv1d', nn.Linear(embed_dim, representation_size)),
        #         ('act', nn.Tanh())
        #     ]))
        # else:
        self.pre_logits = nn.Identity()

        # Classifier head
        self.cls_head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        if fuse_patch:
            self.token_head = FusionLayer(self.num_features, num_token_classes, video_length) if num_token_classes > 0 else nn.Identity()
        else:
            self.token_head = nn.Linear(self.num_features, num_token_classes) if num_token_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def get_classifier(self):
        return self.cls_head, self.token_head

    def reset_classifier(self, num_classes, num_token_classes, fuse_patch=True, global_pool=''):
        self.num_classes = num_classes
        self.num_token_classes = num_token_classes
        if num_classes > 0:
            self.cls_head = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.cls_head.weight, std=0.01)
            nn.init.zeros_(self.cls_head.bias)
        else:
            self.cls_head = nn.Identity()
        if num_token_classes > 0:
            if fuse_patch:
                self.token_head = FusionLayer(self.num_features, num_token_classes, self.video_length)
                nn.init.normal_(self.token_head.head.weight, std=0.01)
                nn.init.zeros_(self.token_head.head.bias)
            else:
                self.token_head = nn.Linear(self.embed_dim, num_token_classes)
                nn.init.normal_(self.token_head.weight, std=0.01)
                nn.init.zeros_(self.token_head.bias)
        else:
            self.token_head = nn.Identity

    def forward_features(self, x):
        """
        :param x: B x L x C x H x W
        :return: x: B x (1+Lhw) x C'
        """
        B, L = x.shape[:2]

        x = x.flatten(start_dim=0, end_dim=1)
        x = self.patch_embed(x)
        x = x.reshape(B, L, x.shape[-2], x.shape[-1]).flatten(start_dim=1, end_dim=2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)  # B x (1+LHW) x C
        cls_logits = self.cls_head(x[:, 0])  # B x C'
        token_logits = self.token_head(x[:, 1:])  # N x L x C''
        return cls_logits, token_logits


def video_vit_small_patch16_224(seq_len, num_classes, num_token_classes, fuse_patch=True, pretrained=True):
    model = VideoTransformer(seq_len, num_token_classes, fuse_patch,
                             patch_size=16, embed_dim=768, depth=8, num_heads=8,
                             mlp_ratio=3., qk_scale=768 ** -0.5)
    if pretrained:
        pretrained_state_dict = vit_small_patch16_224(pretrained=True).state_dict()
        load_pretrained_weights(model, pretrained_state_dict, seq_len)

    model.reset_classifier(num_classes=num_classes, num_token_classes=num_token_classes, fuse_patch=fuse_patch)
    return model


def video_vit_base_patch_224(seq_len, num_classes, num_token_classes, fuse_patch=True, pretrained=True):
    model = VideoTransformer(seq_len, num_token_classes, fuse_patch,
                             patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained:
        pretrained_state_dict = vit_base_patch16_224(pretrained=True).state_dict()
        load_pretrained_weights(model, pretrained_state_dict, seq_len)

    model.reset_classifier(num_classes=num_classes, num_token_classes=num_token_classes, fuse_patch=fuse_patch)
    return model


def video_vit_large_patch_224(seq_len, num_classes, num_token_classes, fuse_patch=True, pretrained=True):
    model = VideoTransformer(seq_len, num_token_classes, fuse_patch,
                             patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained:
        pretrained_state_dict = vit_large_patch16_224(pretrained=True).state_dict()
        load_pretrained_weights(model, pretrained_state_dict, seq_len)

    model.reset_classifier(num_classes=num_classes, num_token_classes=num_token_classes, fuse_patch=fuse_patch)
    return model


def load_pretrained_weights(model, pretrained_state_dict, seq_len):
    # copy everything but pos_embed
    pretrained_pos_embed = pretrained_state_dict['pos_embed']
    del pretrained_state_dict['pos_embed']
    model.load_state_dict(pretrained_state_dict, strict=False)

    # duplicate pos_embed by seq_len.
    model.pos_embed[0, 0].data = pretrained_pos_embed[0, 0].data
    model.pos_embed[0, 1:].data = pretrained_pos_embed[0, 1:].repeat(1, seq_len, 1)


if __name__ == '__main__':
    device = 'cuda'
    x = torch.randn(2, 4, 3, 224, 224).to(device)
    model = VideoTransformer(4, 8, fuse_patch=True, img_size=224, num_classes=6).to(device)
    cls_logits, token_logits = model(x)
    print(cls_logits.shape, token_logits.shape)