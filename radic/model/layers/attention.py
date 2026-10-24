import torch
import torch.nn as nn
from torch import Tensor
from .conv import conv1x1, conv3x3

class AttentionBlock(nn.Module):

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(conv1x1(N, N // 2), nn.ReLU(inplace=True), conv3x3(N // 2, N // 2), nn.ReLU(inplace=True), conv1x1(N // 2, N))
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out
        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())
        self.conv_b = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit(), conv1x1(N, N))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out
