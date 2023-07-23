import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.01

class Net_s10(nn.Module):
    def __init__(self):
        super(Net_s10, self).__init__()
        # Input Block
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) # output_size = 30
        

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
 
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
 
        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) # output_size = 30

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(4,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.conv1(x)
        r1 = self.res1(x)
        x = x+r1
        x = self.conv2(x)
        x = self.conv3(x)
        r2 = self.res2(x)
        x = x+r2
        x = self.maxpool(x)
        x = self.fc(torch.squeeze(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)