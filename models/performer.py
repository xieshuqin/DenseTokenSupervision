# Implement per-token output video performer
# Code is based on
#     https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     https://github.com/yitu-opensource/T2T-ViT/blob/main/models/token_performer.py


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.vision_transformer import Mlp, DropPath, trunc_normal_
from performer_pytorch import SelfAttention

from models.transformer import VideoTransformer


"""Attention from VisionTransformer"""
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerformerAttention(nn.Module):
    # A warpper class for SelfAttention module to make it compatible with VisionTransformer
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.layer = SelfAttention(
            dim,
            heads=num_heads,
            dim_head=None,
            causal=False,
            local_heads=0,
            dropout=attn_drop,
            nb_features=256,
        )

    def forward(self, x):
        return self.layer(x)


class PerformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PerformerAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VideoPerformer(VideoTransformer):
    def __init__(self,
                 *args,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=False,
                 **kwargs):
        super().__init__(*args,
                         depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                         **kwargs)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PerformerBlock(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])


        if pretrained:
            patch_size=kwargs['patch_size']
            embed_dim=kwargs['embed_dim']
            transformer=None
            if patch_size==16 and embed_dim==768 and depth==8 and mlp_ratio==3. and qkv_bias==False and norm_layer==nn.LayerNorm:
                from timm.models.vision_transformer import vit_small_patch16_224
                transformer=lambda:vit_small_patch16_224(True)
            assert transformer is not None
            # from timm.models.vision_transformer import vit_base_patch16_224
            # transformer=vit_base_patch16_224(pretrained=True)
            transformer = transformer()
            model=self
            for old_key in transformer.state_dict():
                if old_key in ['head.weight','head.bias']:
                    continue
                elif old_key == 'pos_embed':
                    new_key=old_key
                    model.state_dict()[new_key][0,0]=transformer.state_dict()[old_key][0,0]
                    num_patch = transformer.state_dict()[old_key].size(1) - 1
                    video_length = (model.state_dict()[new_key].size(1)-1) // num_patch
                    for i in range(video_length):
                        model.state_dict()[new_key][0,1+i*num_patch:1+(i+1)*num_patch] = transformer.state_dict()[old_key][0,1:]
                    print(new_key,'<-',old_key)
                elif 'attn' not in old_key:
                    new_key=old_key
                    model.state_dict()[new_key][:]=transformer.state_dict()[old_key]
                    print(new_key,'<-',old_key)
                elif 'proj' in old_key:
                    new_key=old_key.replace('proj','layer.to_out')
                    model.state_dict()[new_key][:]=transformer.state_dict()[old_key]
                    print(new_key,'<-',old_key)
                else:
                    new_key=old_key.replace('qkv','layer.to_q')
                    model.state_dict()[new_key][:]=transformer.state_dict()[old_key][:embed_dim]
                    print(new_key,'<-',old_key+'[:%d]'%embed_dim)
                    new_key=new_key.replace('weight','bias')
                    model.state_dict()[new_key][:]=0
                    print(new_key,'<-',0)
                    new_key=old_key.replace('qkv','layer.to_k')
                    model.state_dict()[new_key][:]=transformer.state_dict()[old_key][embed_dim:2*embed_dim]
                    print(new_key,'<-',old_key+'[%d:%d]'%(2*embed_dim,embed_dim))
                    new_key=new_key.replace('weight','bias')
                    model.state_dict()[new_key][:]=0
                    print(new_key,'<-',0)
                    new_key=old_key.replace('qkv','layer.to_v')
                    model.state_dict()[new_key][:]=transformer.state_dict()[old_key][2*embed_dim:]
                    print(new_key,'<-',old_key+'[%d:]'%(2*embed_dim))
                    new_key=new_key.replace('weight','bias')
                    model.state_dict()[new_key][:]=0
                    print(new_key,'<-',0)


if __name__ == '__main__':
    # test VideoPerformer
    import torchprof
    device = 'cuda'
    N, L = 4, 8
    x = torch.randn(N, L, 3, 224, 224).to(device)
    cls_label = torch.randint(6, (N,), dtype=torch.long).to(device)
    token_label = torch.randint(8, (N, L), dtype=torch.long).to(device)
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,
        qkv_bias=False, norm_layer=nn.LayerNorm)
    model = VideoPerformer(pretrained=True,video_length=L, num_token_classes=8, fuse_patch=True, img_size=224, num_classes=6, **model_kwargs).to(device)
    # with torchprof.Profile(model, use_cuda=False, profile_memory=True) as prof:
    cls_logits, token_logits = model(x)
    loss_cls = F.cross_entropy(cls_logits, cls_label)
    loss_token = F.cross_entropy(token_logits.flatten(start_dim=0, end_dim=1), token_label.flatten())
    loss = loss_cls + loss_token
    loss.backward()
    # print(prof.display(show_events=False))
