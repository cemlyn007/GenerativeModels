import sys

import torch
import torch.nn as nn

print(sys.path)

from utils.module_helpers import PrintTensors


class LinearVAE(nn.Module):
    def __init__(self, latent_dim):
        super(LinearVAE, self).__init__()

        # Encoding Material
        self.encoder = nn.Sequential(
            PrintTensors("Input"),
            nn.Linear(784, 980),
            nn.ReLU(inplace=True),
            PrintTensors("1"),
            nn.Linear(980, 1024),
            nn.ReLU(inplace=True),
            PrintTensors("2"),
            nn.Linear(1024, 1280),
            nn.ReLU(inplace=True),
            PrintTensors("Encoded"),
        )

        self.en_mu_lin = nn.Linear(1280, latent_dim)
        self.en_logvar_lin = nn.Linear(1280, latent_dim)

        # Decoding Material
        self.decoder = nn.Sequential(
            PrintTensors("Input"),
            nn.Linear(latent_dim, 1280),
            nn.ReLU(inplace=True),
            PrintTensors("1"),
            nn.Linear(1280, 1024),
            nn.ReLU(inplace=True),
            PrintTensors("2"),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
            PrintTensors("Output"),
        )

    def encode(self, x):
        x = self.encoder(x.view((-1, 784)))
        return self.en_mu_lin(x), self.en_logvar_lin(x)

    def reparametrize(self, mu, logvar):
        return mu + torch.exp(logvar * 0.5) * torch.randn_like(logvar)

    def decode(self, z):
        x = self.decoder(z)
        return x.view((-1, 1, 28, 28))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar


class ConvVAE(nn.Module):

    def __init__(self, latent_dim: int, amp=1):
        super(ConvVAE, self).__init__()

        # Encoding Material
        self.encoder = nn.Sequential(
            PrintTensors("Input"),
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            PrintTensors("1"),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            PrintTensors("2"),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            PrintTensors("3"),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            PrintTensors("Encoded"),
        )

        self.en_mu_lin = nn.Linear(64, latent_dim)
        self.en_logvar_lin = nn.Linear(64, latent_dim)

        # Decoding Material
        self.decoder = nn.Sequential(
            PrintTensors("Input"),
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            PrintTensors("1"),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            PrintTensors("2"),
            nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            PrintTensors("3"),
            nn.ConvTranspose2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            PrintTensors("4"),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            PrintTensors("5"),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            PrintTensors("Output"),
        )

    def encode(self, x):
        x = self.encoder(x).flatten(start_dim=1)
        return self.en_mu_lin(x), self.en_logvar_lin(x)

    def reparametrize(self, mu, logvar):
        return mu + torch.exp(logvar * 0.5) * torch.randn_like(logvar)

    def decode(self, z):
        x = self.decoder(z)
        return x.view((-1, 1, 28, 28))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = self.decode(z)
        return x, mu, logvar
