import torch.nn as nn
import torch
from collections import OrderedDict

class ConvBnReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


def conv3d(in_channels, out_channels, kernel_size=3, stride=1):
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)




class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class InteractiveHNN(nn.Module):
    """
    adapted version of Duanmu et al. (MICCAI, 2020)
    https://link.springer.com/chapter/10.1007%2F978-3-030-59713-9_24
    """

    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
    ) -> None:

        super().__init__()

        # ResNet
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

        layers = [
            ("aux_base", nn.Linear(ndim_non_img, 8, bias=False)),
            ("aux_relu", nn.ReLU()),
            # ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_1", nn.Linear(8, n_basefilters, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

        self.aux_2 = nn.Linear(n_basefilters, n_basefilters, bias=False)
        self.aux_3 = nn.Linear(n_basefilters, 2 * n_basefilters, bias=False)
        self.aux_4 = nn.Linear(2 * n_basefilters, 4 * n_basefilters, bias=False)

    @property
    def input_names(self):
        return ("image", "tabular")

    @property
    def output_names(self):
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)

        attention = self.aux(tabular)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block1(out)

        attention = self.aux_2(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block2(out)

        attention = self.aux_3(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block3(out)

        attention = self.aux_4(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block4(out)

        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out