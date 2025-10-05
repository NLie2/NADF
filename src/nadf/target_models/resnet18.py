from functools import partial
from typing import Type

import torch
import torch.nn as nn
from torch import Tensor

from nadf.utils import LRConv2d

# from Gemini


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
        relu_inplace: bool = True,
        padding_mode: str = "zeros",
        conv_fnc=nn.Conv2d,
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = conv_fnc(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv_fnc(
            out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int = 1000,
        get_learned_repr: bool = False,
        get_all_repr_norms: int = 0,
        relu_inplace: bool = True,
        padding_mode: str = "zeros",
        width=64,
        normalize_logits=False,
        layer_rank=-1,
        **kwargs,
    ) -> None:
        super(ResNet18, self).__init__()
        assert get_all_repr_norms in [0, 1, 2]
        self.conv_fnc = nn.Conv2d if layer_rank == -1 else partial(LRConv2d, rank=layer_rank)
        block = partial(block, conv_fnc=self.conv_fnc)
        self.normalize_logits = normalize_logits
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = width
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        # Do not implement low-rankness in the input layer.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            padding_mode=padding_mode,
        )
        self.get_learned_repr = get_learned_repr
        self.repr_norm = get_all_repr_norms
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], relu_inplace=relu_inplace, padding_mode=padding_mode)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, relu_inplace=relu_inplace, padding_mode=padding_mode
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, relu_inplace=relu_inplace, padding_mode=padding_mode
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, relu_inplace=relu_inplace, padding_mode="zeros"
        )  # hack

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1,
        relu_inplace=False,
        padding_mode="zeros",
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                self.conv_fnc(
                    self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                self.expansion,
                downsample,
                relu_inplace=relu_inplace,
                padding_mode=padding_mode,
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    expansion=self.expansion,
                    relu_inplace=relu_inplace,
                    padding_mode=padding_mode,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if hasattr(self, "repr_norm") and self.repr_norm:
            self.total_norm = 0
        x = self.layer1(x)
        if hasattr(self, "repr_norm") and self.repr_norm:
            self.total_norm += torch.norm(x, p=self.repr_norm)
        x = self.layer2(x)
        if hasattr(self, "repr_norm") and self.repr_norm:
            self.total_norm += torch.norm(x, p=self.repr_norm)
        x = self.layer3(x)
        if hasattr(self, "repr_norm") and self.repr_norm:
            self.total_norm += torch.norm(x, p=self.repr_norm)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if hasattr(self, "repr_norm") and self.repr_norm:
            self.total_norm += torch.norm(x, p=self.repr_norm)
        if hasattr(self, "get_learned_repr") and self.get_learned_repr:
            self.learned_repr = x
        x = self.fc(x)
        if getattr(self, "normalize_logits", False):
            x = x / x.norm(dim=1, keepdim=True)
        return x


class MaskedResNet(nn.Module):
    def __init__(
        self, original_model, layer_index, mask=None, activation_exponent=None, mean_replacement=None, dropout=None
    ):
        super(MaskedResNet, self).__init__()
        self.model = original_model
        self.mask = mask
        self.activation_exponent = activation_exponent
        self.mean_replacement = mean_replacement
        if dropout is not None:
            self.do_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if hasattr(self, "do_layer"):
            x = self.do_layer(x)
        if self.mask is not None:
            x = x * self.mask
        if self.mean_replacement is not None:
            x = x + (1 - self.mask) * self.mean_replacement
        if self.activation_exponent is not None:
            x = x**self.activation_exponent

        x = self.model.fc(x)

        return x


ModifiedResNet = MaskedResNet
