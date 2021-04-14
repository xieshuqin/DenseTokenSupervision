import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet


class CustomResNet(resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 2).transpose(1, 2)
        x = self.fc(x)

        return x


def load_pretrained(model, pretrained_model):
    state_dict = pretrained_model.state_dict()
    model.load_state_dict(state_dict)


def resnet18(pretrained=False, **kwargs):
    model = CustomResNet(resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_model = resnet.resnet18(pretrained=True)
        load_pretrained(model, pretrained_model)
    return model


def resnet34(pretrained=False, **kwargs):
    model = CustomResNet(resnet.BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = resnet.resnet34(pretrained=True)
        load_pretrained(model, pretrained_model)
    return model


def resnet50(pretrained=False, **kwargs):
    model = CustomResNet(resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = resnet.resnet50(pretrained=True)
        load_pretrained(model, pretrained_model)
    return model


def resnet101(pretrained=False, **kwargs):
    model = CustomResNet(resnet.Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_model = resnet.resnet101(pretrained=True)
        load_pretrained(model, pretrained_model)
    return model


if __name__ == '__main__':
    model = resnet50(pretrained=True).cuda()
    x = torch.randn(4, 3, 224, 224, device='cuda')
    x = model(x)