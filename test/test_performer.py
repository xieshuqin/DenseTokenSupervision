import torch
import torch.nn as nn
import torch.nn.functional as F
import torchprof

from timm.models.vision_transformer import Attention, VisionTransformer
from models.transformer import VideoViT
from models.performer import PerformerAttention, PerformerBlock, VideoPerformer


def test_attention():
    device = 'cuda'
    N, L, C = 3, 1024, 512
    x = torch.randn(N, L, C).to(device)
    attention = Attention(512, 8).to(device)
    linAttention = PerformerAttention(512, 8).to(device)

    out1 = attention(x)
    out2 = linAttention(x)
    print(out1.shape)
    print(out2.shape)


def test_performer():
    device = 'cuda'
    N, L = 8, 24
    x = torch.randn(N, L, 3, 224, 224).to(device)
    cls_label = torch.randint(6, (N,), dtype=torch.long).to(device)
    token_label = torch.randint(8, (N, L), dtype=torch.long).to(device)
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,
        qkv_bias=False, norm_layer=nn.LayerNorm)
    model = VideoPerformer(video_length=L, num_token_classes=8, fuse_patch=True, img_size=224, num_classes=6,
                           **model_kwargs).to(device)
    with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
        cls_logits, token_logits = model(x)
        loss_cls = F.cross_entropy(cls_logits, cls_label)
        loss_token = F.cross_entropy(token_logits.flatten(start_dim=0, end_dim=1), token_label.flatten())
        loss = loss_cls + loss_token
        loss.backward()
    print(prof.display(show_events=False))


def test_resnet_bottlenect():
    from torchvision.models import resnet50
    device = 'cuda'
    model = resnet50(True).to(device)
    N, L = 16, 32
    x = torch.randn(N*L, 3, 224, 224).cuda()
    model.eval()
    with torch.no_grad():
        out = model(x)