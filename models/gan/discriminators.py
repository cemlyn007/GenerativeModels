import torch.nn as nn

from utils.module_helpers import PrintTensors


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.judge = nn.Sequential(
            nn.Conv2d(3, 256, 3, 1, 0, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, 3, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, 1, 0, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, 2, 0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, 1, 0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, 2, 0, bias=True),
            nn.Sigmoid()
        )

    def discriminator(self, x):
        return self.judge(x)

    def forward(self, x):
        return self.discriminator(x).view(-1, 1).squeeze(1)
