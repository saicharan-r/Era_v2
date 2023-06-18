import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )  # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        )  # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )  # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )  # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )  # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )  # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )  # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(0.1)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock5(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0
            nn.ReLU(),
        )  # output_size = 26

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0
            nn.ReLU(),  # output_size = 24
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_1
            nn.ReLU(),  # output_size = 22
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # trans1_pool0 # output_size = 11
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # trans1_conv0
            nn.ReLU(),  # output_size = 11
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0
            nn.ReLU(),  # output_size = 9
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_1
            nn.ReLU(),  # output_size = 7
        )

        # Output
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv3_0
            nn.ReLU(),  # output_size = 7
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(7, 7),
                padding=0,
                bias=False,
            ),  # conv3_1 # output_size = 7x7x10 | 7x7x10x10 | 1x1x10
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
