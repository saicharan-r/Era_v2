import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class NormalizationMethod(Enum):
    BATCH = 1
    LAYER = 2
    GROUP = 3


def normalizer(
    method: NormalizationMethod,
    out_channels: int,
) -> nn.BatchNorm2d | nn.GroupNorm:
    switcher = {
        NormalizationMethod.BATCH: nn.BatchNorm2d(out_channels),
        NormalizationMethod.LAYER: nn.GroupNorm(1, out_channels),
        NormalizationMethod.GROUP: nn.GroupNorm(8, out_channels),
    }
    return switcher.get(method, ValueError("Invalid NormalizationMethod"))

class Model(nn.Module):
  def __init__(self,normalization_method: NormalizationMethod):
    super(Model,self).__init__()
    self.normalization_method = normalization_method

    self.ConvBlock1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16,kernel_size=(3,3),padding=1,bias=False),
        normalizer(normalization_method,16),
        nn.ReLU(),
        nn.Dropout(0.05)
    )#output size = 32
    self.ConvBlock2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=16,kernel_size=(3,3),padding=1,bias=False),
        normalizer(normalization_method,16),
        nn.ReLU(),
        nn.Dropout(0.05)
    )#output size = 32
    self.trans1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # output_size = 32
            nn.MaxPool2d(2, 2)
    )#output size = 16
    self.ConvBlock3 = nn.Sequential(
        nn.Conv2d(in_channels=8, out_channels=16,kernel_size=(3,3),padding=1,bias=False),
        normalizer(normalization_method,16),
        nn.ReLU(),
        nn.Dropout(0.05)
    )#output size = 16
    self.ConvBlock4 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=16,kernel_size=(3,3),padding=1,bias=False),
        normalizer(normalization_method,16),
        nn.ReLU(),
        nn.Dropout(0.05)
    )#output size = 16
    self.trans2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # output_size = 28
            nn.MaxPool2d(2, 2),  # output_size = 12
    )#output size = 8
    self.ConvBlock5 = nn.Sequential(
        nn.Conv2d(in_channels=8, out_channels=16,kernel_size=(3,3),padding=1,bias=False),
        normalizer(normalization_method,16),
        nn.ReLU(),
        nn.Dropout(0.05)
    )#output size = 8
    self.ConvBlock6 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32,kernel_size=(3,3),padding=1,bias=False),
        normalizer(normalization_method,32),
        nn.ReLU(),
        nn.Dropout(0.05)
    )#output size = 8
    self.ConvBlock7 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32,kernel_size=(3,3),padding=1,bias=False),
        normalizer(normalization_method,64),
        nn.ReLU(),
        nn.Dropout(0.05)
    )#output size = 8
    self.gap=nn.Sequential(
        nn.AdaptiveAvgPool2d(1)
    )
    self.out=nn.Sequential(
         nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # output_size = 28
    )


  def forward(self, x):
        x = self.ConvBlock1(x)
        x = x + self.ConvBlock2(x)
        x = self.trans1(x)
        x = self.ConvBlock3(x)
        x = x + self.ConvBlock4(x)
        x = self.trans2(x)
        x = self.ConvBlock5(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.gap(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
