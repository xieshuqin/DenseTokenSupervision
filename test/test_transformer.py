import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import VisionTransformer, vit_small_patch16_224, vit_base_patch16_224
from models.transformer import VideoTransformer
# from models.performer import VideoPerformer
from models.new_performer import VideoPerformer


from datasets import build_dataloader


def test_transformer():
    dataloader = build_dataloader('finegym', split='train', batch_size=4, num_frames_per_video=1)
    model = vit_small_patch16_224(pretrained=False)
    model.reset_classifier(num_classes=4)
    nn.init.normal_(model.head.weight, std=0.01)
    nn.init.zeros_(model.head.bias)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=4e-5)

    step = 0
    for (x, y_frame, y_video) in dataloader:
        x = x.squeeze(1).cuda()
        y_video = y_video.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits, y_video)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        print(f'loss at step {step} is {loss.item():.4f}')



def test_video_transformer():
    dataloader = build_dataloader('finegym', split='train', batch_size=4, num_frames_per_video=8)
    pretrained_vit = vit_small_patch16_224(pretrained=True)
    model = VideoTransformer(8, 288, True, patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., qk_scale=768**-0.5)

    pretrained_state_dict = pretrained_vit.state_dict()
    pretrained_pos_embed = pretrained_state_dict['pos_embed']
    del pretrained_state_dict['pos_embed']
    model.load_state_dict(pretrained_state_dict, strict=False)
    model.pos_embed[0,0].data = (pretrained_pos_embed[0,0].data)
    for i in range(8):
        model.pos_embed[0,1+14*14*(i):1+14*14*(i+1),:].data = pretrained_pos_embed[0,1:].data

    model.reset_classifier(num_classes=4, num_token_classes=288)
    nn.init.normal_(model.cls_head.weight, std=0.01)
    nn.init.zeros_(model.cls_head.bias)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=4e-5)

    step = 0
    for (x, y_frame, y_video) in dataloader:
        x = x.cuda()
        y_frame = y_frame.cuda()
        y_video = y_video.cuda()
        video_logits, frame_logits = model(x)
        loss_video = F.cross_entropy(video_logits, y_video)
        loss_frame = F.cross_entropy(frame_logits.flatten(0, 1), y_frame.flatten())
        loss = loss_video + loss_frame

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        print(f'Step {step}, loss {loss.item():.4f}, loss_video {loss_video.item():.4f}, loss_frame {loss_frame.item():.4f}')


def test_video_performer():
    dataloader = build_dataloader('finegym', split='train', batch_size=4, num_frames_per_video=1)
    pretrained_vit = vit_base_patch16_224(pretrained=True)
    model = VideoPerformer(1, 288, True,
                           patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           )
    model.load_state_dict(pretrained_vit.state_dict(), strict=False)
    model.reset_classifier(num_classes=4, num_token_classes=288)
    nn.init.normal_(model.cls_head.weight, std=0.01)
    nn.init.zeros_(model.cls_head.bias)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=4e-5)

    step = 0
    for (x, y_frame, y_video) in dataloader:
        x = x.cuda()
        y_video = y_video.cuda()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y_video)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        print(f'loss at step {step} is {loss.item():.4f}')


if __name__ == '__main__':
    # test_video_performer()
    test_video_transformer()