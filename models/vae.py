import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio

import sys

print(sys.path)

from utils.module_helpers import PrintTensors


def show(img):
    if torch.cuda.is_available():
        img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# Modify the denorm function in case you need to do any output transformation when visualizing your images
def interpolate(z1, z2, num=11):
    Z = np.zeros((z1.shape[0], num))
    for i in range(z1.shape[0]):
        Z[i, :] = np.linspace(z1[i], z2[i], num)
    return Z


def denorm_for_tanh(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def denorm_for_sigmoid(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class LinearVAE(nn.Module):
    def __init__(self, latent_dim):
        super(LinearVAE, self).__init__()

        # Encoding Material
        self.encoder = Sequential(
            PrintTensors("Input"),
            Linear(784, 980),
            ReLU(inplace=True),
            PrintTensors("1"),
            Linear(980, 1024),
            ReLU(inplace=True),
            PrintTensors("2"),
            Linear(1024, 1280),
            ReLU(inplace=True),
            PrintTensors("Encoded"),
        )

        self.en_mu_lin = Linear(1280, latent_dim)
        self.en_logvar_lin = Linear(1280, latent_dim)

        # Decoding Material
        self.decoder = Sequential(
            PrintTensors("Input"),
            Linear(latent_dim, 1280),
            ReLU(inplace=True),
            PrintTensors("1"),
            Linear(1280, 1024),
            ReLU(inplace=True),
            PrintTensors("2"),
            Linear(1024, 784),
            Sigmoid(),
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

    def __init__(self, amp=1):
        super(ConvVAE, self).__init__()

        # Encoding Material
        self.encoder = Sequential(
            PrintTensors("Input"),
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2),
            ReLU(inplace=True),
            PrintTensors("1"),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            ReLU(inplace=True),
            PrintTensors("2"),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=2),
            ReLU(inplace=True),
            PrintTensors("3"),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=2, stride=1),
            ReLU(inplace=True),
            PrintTensors("Encoded"),
        )

        self.en_mu_lin = Linear(64, latent_dim)
        self.en_logvar_lin = Linear(64, latent_dim)

        # Decoding Material
        self.decoder = Sequential(
            PrintTensors("Input"),
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=256, kernel_size=2, stride=1),
            ReLU(inplace=True),
            PrintTensors("1"),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            ReLU(inplace=True),
            PrintTensors("2"),
            nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=4, stride=2),
            ReLU(inplace=True),
            PrintTensors("3"),
            nn.ConvTranspose2d(in_channels=96, out_channels=64, kernel_size=4, stride=2),
            ReLU(inplace=True),
            PrintTensors("4"),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1),
            Sigmoid(),
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


if __name__ == "__main__":
    from torch.nn import Conv2d, ConvTranspose3d, Linear, ReLU, Sequential, Sigmoid
    from torch.nn.modules.pooling import MaxPool2d

    # device selection
    GPU = True
    device_idx = 0
    if GPU:
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    torch.manual_seed(0)

    if not os.path.exists('../CW_VAE/MNIST'):
        os.makedirs('../CW_VAE/MNIST')

    """## Hyper-parameter selection"""

    # *CODE FOR PART 1.1 IN THIS CELL*

    ### Choose the number of epochs, the learning rate and the batch size
    num_epochs = 20
    learning_rate = 5e-5  # 1e-3
    batch_size = 512
    ### Choose a value for the size of the latent space
    latent_dim = 10

    ###

    # Define here the any extra hyperparameters you used.
    beta = 2  # .75 WAS 0.5
    ###

    # Modify this line if you need to do any input transformations (optional).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,)),
    ])

    denorm = denorm_for_sigmoid

    """## Data loading"""

    train_dat = datasets.MNIST(
        "../data/", train=True, download=True, transform=transform
    )
    test_dat = datasets.MNIST("../data/", train=False, transform=transform)

    loader_train = DataLoader(train_dat, batch_size, shuffle=True)
    loader_test = DataLoader(test_dat, batch_size, shuffle=False)

    sample_inputs, _ = next(iter(loader_test))
    fixed_input = sample_inputs[:32, :, :, :]

    save_image(fixed_input, '../CW_VAE/MNIST/image_original.png')

    model = ConvVAE(amp=1).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    print(model)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # *CODE FOR PART 1.1b IN THIS CELL*
    def loss_function_VAE(recon_x, x, mu, logvar, beta):
        BCE = torch.sum(F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784),
                                               reduction='none'), dim=1)
        Expected_BCE = torch.mean(BCE)
        Batch_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = Expected_BCE + beta * Batch_KLD
        return loss, Expected_BCE, Batch_KLD


    epoch_loss = []
    reconstruction_list = []
    KLD_list = []
    test_loss_lst = []
    test_BCE_lst = []
    test_KLD_lst = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        reconstruction_loss = 0
        KLD_loss = 0

        for batch_idx, data in enumerate(loader_train):
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            # forward
            recon_batch, mu, logvar = model(img)
            loss, BCE, KLD = loss_function_VAE(recon_batch, img, mu, logvar, beta)
            # backward
            loss.backward()
            train_loss += loss.item()
            reconstruction_loss += BCE.item()
            KLD_loss += KLD.item()
            # print(f"Debug Train: Avg Loss: {loss.item()}, Reconstruction Loss: {BCE.item()} and KLD: {KLD.item()}")
            optimizer.step()
        # print out losses and save reconstructions for every epoch
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, train_loss))
        recon = denorm(model(fixed_input.to(device))[0])
        save_image(recon.float(), '../CW_VAE/MNIST/reconstructed_epoch_{}.png'.format(epoch))
        print(f"Train: Loss: {train_loss}, Reconstruction Loss: {reconstruction_loss} and KLD Loss: {KLD_loss}")
        epoch_loss.append(train_loss)
        reconstruction_list.append(reconstruction_loss)
        KLD_list.append(KLD_loss)

        model.eval()
        test_loss = 0
        test_BCE = 0
        test_KLD = 0
        with torch.no_grad():
            for img, _ in loader_test:
                img = img.to(device)
                recon_batch, mu, logvar = model(img)
                loss, BCE, KLD = loss_function_VAE(recon_batch, img, mu, logvar, beta)
                test_loss += loss
                test_BCE += BCE
                test_KLD += KLD
            # reconstruct and save the last batch
            recon_batch = model(recon_batch.to(device))
            img = denorm(img.cpu())

        test_loss_lst.append(test_loss)
        test_BCE_lst.append(test_BCE)
        test_KLD_lst.append(test_KLD)
        print(f"Test: Total Loss: {test_loss}, Reconstruction Loss: {test_BCE} and KLD: {test_KLD}")

    images = []
    for epoch in range(num_epochs):
        images.append(imageio.imread('../CW_VAE/MNIST/reconstructed_epoch_{}.png'.format(epoch)))
    imageio.mimsave('/VAE_movie.gif', images)

    # save the model
    torch.save(model.state_dict(), '../CW_VAE/MNIST/VAE_model.pth')
