import torch.nn as nn

from utils.module_helpers import PrintTensors


class Generator(nn.Module):
    def __init__(self, latent_vector_size: int):
        super(Generator, self).__init__()

        self.decoder = nn.Sequential(
            PrintTensors("Input"),
            nn.ConvTranspose2d(in_channels=latent_vector_size, out_channels=256, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PrintTensors("1"),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PrintTensors("2"),
            nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            PrintTensors("3"),
            nn.ConvTranspose2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PrintTensors("4"),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PrintTensors("5"),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
            PrintTensors("Output"),
        )

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, z):
        return self.decode(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.judge = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(96, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def discriminator(self, x):
        out = self.judge(x)
        return out

    def forward(self, x):
        outs = self.discriminator(x)
        return outs.view(-1, 1).squeeze(1)
