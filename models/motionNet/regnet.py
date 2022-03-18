'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .preresnet import BasicBlock, Bottleneck


__all__ = ['RegNet', 'regnet']


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class RegNet(nn.Module):
    def __init__(self, block, channel_in, num_classes):
        super(RegNet, self).__init__()
        layers = [2, 4, 6, 3]
        self.inplanes = 32
        self.conv1 = nn.Conv2d(channel_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, 32, layers[0], stride=2)
        self.layer2 = self._make_residual(block, 64, layers[1], stride=2)
        self.layer3 = self._make_residual(block, 128, layers[2], stride=2)
        self.layer4 = self._make_residual(block, 256, layers[3], stride=2)
        self.glbpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes-1)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.glbpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)

        return out


def regnet(**kwargs):
    if kwargs['color_mode'] == 'RGB':
        channel_in = 3
    else:
        channel_in = 1
    model = RegNet(Bottleneck, channel_in, kwargs['num_classes'])

    return model
