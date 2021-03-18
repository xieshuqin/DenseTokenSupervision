# Implement per-token output transformer
# Code is based on
#     https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.vision_transformer import VisionTransformer, trunc_normal_


class VisionTransformerTmp(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


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


class VideoViT(VisionTransformer):
    def __init__(self, video_length, num_token_classes, fuse_patch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_length = video_length
        self.num_token_classes = num_token_classes
        self.fuse_patch = fuse_patch

        num_patches_per_frame, embed_dim = self.patch_embed.num_patches, self.embed_dim
        token_length = video_length * num_patches_per_frame
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+token_length, embed_dim))

        #  Representation layer
        if hasattr(self, 'num_features'):
            representation_size = self.num_features
            self.pre_logits = nn.Sequential(OrderedDict([
                ('conv1d', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
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
        self.cls_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if fuse_patch:
            self.token_head = FusionLayer(self.num_features, num_token_classes, self.video_length) if num_token_classes > 0 else nn.Identity()
        else:
            self.token_head = nn.Linear(self.embed_dim, num_token_classes) if num_token_classes > 0 else nn.Identity()

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


if __name__ == '__main__':
    device = 'cuda'
    x = torch.randn(2, 4, 3, 224, 224).to(device)
    model = VideoViT(4, 8, fuse_patch=True, img_size=224, num_classes=6).to(device)
    cls_logits, token_logits = model(x)
    print(cls_logits.shape, token_logits.shape)