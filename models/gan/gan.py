import torch.nn as nn

from utils.module_helpers import PrintTensors


class Generator(nn.Module):
    def __init__(self, latent_vector_size: int):
        super(Generator, self).__init__()

        self.decoder = nn.Sequential(
            PrintTensors("Input"),
            nn.ConvTranspose2d(latent_vector_size, 1024, 4, 2, 0, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            PrintTensors("1"),
            nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            PrintTensors("2"),
            nn.ConvTranspose2d(512, 256, 2, 2, 0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            PrintTensors("3"),
            nn.ConvTranspose2d(256, 128, 3, 1, 0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            PrintTensors("4"),
            nn.ConvTranspose2d(128, 64, 2, 2, 0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            PrintTensors("5"),
            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=True),
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
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def discriminator(self, x):
        out = self.judge(x)
        return out

    def forward(self, x):
        outs = self.discriminator(x)
        return outs.view(-1, 1).squeeze(1)
