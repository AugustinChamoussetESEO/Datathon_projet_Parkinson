import random
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F, Parameter
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.ops import Permute, StochasticDepth, Conv2dNormActivation

import numpy as np

from facial.pain.attention import AttentionBlock


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(
            self,
            dim,
            layer_scale: float,
            stochastic_depth_prob: float,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
            self,
            input_channels: int,
            out_channels: Optional[int],
            num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                        SELayer(cnf.out_channels)
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.pre_classification = nn.Sequential(
            norm_layer(lastconv_output_channels),
            nn.Flatten(1)
        )

        self.attention = AttentionBlock(lastconv_output_channels, 1)
        self.fc = nn.Linear(lastconv_output_channels, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x: Tensor):
        x = self.stn(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.pre_classification(x)
        x, h = self.attention(x)
        x = self.fc(x)

        att_loss = torch.mean(torch.abs(h))

        return x, att_loss


def get_convnext(num_classes, model_size='tiny'):
    if model_size == 'tiny':
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 9),
            CNBlockConfig(768, None, 3),
        ]

        url = 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth'
    elif model_size == 'small':
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 27),
            CNBlockConfig(768, None, 3),
        ]

        url = "https://download.pytorch.org/models/convnext_small-0c510722.pth"
    elif model_size == 'base':
        block_setting = [
            CNBlockConfig(128, 256, 3),
            CNBlockConfig(256, 512, 3),
            CNBlockConfig(512, 1024, 27),
            CNBlockConfig(1024, None, 3),
        ]

        url = "https://download.pytorch.org/models/convnext_base-6075fbad.pth"
    elif model_size == 'large':
        block_setting = [
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 3),
            CNBlockConfig(768, 1536, 27),
            CNBlockConfig(1536, None, 3),
        ]

        url = "https://download.pytorch.org/models/convnext_large-ea097f82.pth"

    net = ConvNeXt(
        block_setting,
        num_classes=num_classes,
        stochastic_depth_prob=0.1
    )

    state_dict = load_state_dict_from_url(url=url)
    net.load_state_dict(state_dict, strict=False)

    return net


class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        self.vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attention = AttentionBlock(512, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.attention(x)
        x = self.fc(x)

        return x
