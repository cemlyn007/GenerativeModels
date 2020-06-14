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


def denorm(x, channels=None, w=None, h=None, resize=False):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x


def show(img):
    if torch.cuda.is_available():
        img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# device selection
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

# # We set a random seed to ensure that your results are reproducible.
# if torch.cuda.is_available():
#     torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

if not os.path.exists('../CW_DCGAN'):
    os.makedirs('../CW_DCGAN')

"""### Data loading"""

batch_size = 64  # change that 256
NUM_TRAIN = 49000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_dir = '../datasets'

cifar10_train = datasets.CIFAR10(data_dir, train=True, download=True,
                                 transform=transform)
cifar10_val = datasets.CIFAR10(data_dir, train=True, download=True,
                               transform=transform)
cifar10_test = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=transform)

loader_train = DataLoader(cifar10_train, batch_size=batch_size,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(cifar10_val, batch_size=batch_size,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
loader_test = DataLoader(cifar10_test, batch_size=batch_size)

"""### Hyper-parameter selection"""

# *CODE FOR PART 2.1 IN THIS CELL*

### Choose the number of epochs, the learning rate
#   and the size of the Generator's input noise vetor.
num_epochs = 100
learning_rate = 0.0002
latent_vector_size = 100


###

# Define here other hyperparameters that you used.

# *CODE FOR PART 2.1 IN THIS CELL*

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_size, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def decode(self, z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x = self.decoder(z)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        return x

    def forward(self, z):
        return self.decode(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
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
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def discriminator(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        out = self.judge(x)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        return out

    def forward(self, x):
        outs = self.discriminator(x)
        return outs.view(-1, 1).squeeze(1)


"""### Initialize Model and print number of parameters

You can use method `weights_init` to initialize the weights of the Generator and Discriminator networks. Otherwise, implement your own initialization, or do not use at all. You will not be penalized for not using initialization.
"""


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


use_weights_init = True

model_G = Generator().to(device)
if use_weights_init:
    model_G.apply(weights_init)
params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
print("Total number of parameters in Generator is: {}".format(params_G))
print(model_G)
print('\n')

model_D = Discriminator().to(device)
if use_weights_init:
    model_D.apply(weights_init)
params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
print("Total number of parameters in Discriminator is: {}".format(params_D))
print(model_D)
print('\n')

print("Total number of parameters is: {}".format(params_G + params_D))

criterion = nn.BCELoss(reduction='mean')


def loss_function(out, label):
    loss = criterion(out, label)
    return loss


"""### Choose and initialize optimizers"""

# setup optimizer
# You are free to add a scheduler or change the optimizer if you want. We chose one for you for simplicity.
beta1 = 0.45
optimizerD = torch.optim.Adam(model_D.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(model_G.parameters(), lr=learning_rate, betas=(beta1, 0.999))

"""### Define fixed input vectors to monitor training and mode collapse."""

fixed_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
real_label = 1
fake_label = 0

"""### Train"""

# Commented out IPython magic to ensure Python compatibility.
train_losses_G = []
train_losses_D = []

for epoch in range(num_epochs):

    train_loss_D = 0
    train_loss_G = 0
    for i, data in enumerate(loader_train, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        model_D.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = model_D(real_cpu)
        errD_real = loss_function(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
        fake = model_G(noise)
        label.fill_(fake_label)
        output = model_D(fake.detach())
        errD_fake = loss_function(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        train_loss_D += errD.item()

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        model_G.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = model_D(fake)
        errG = loss_function(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        train_loss_G += errG.item()
        optimizerG.step()

        # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
        # #               % (epoch, num_epochs, i, len(loader_train),
        # errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if epoch == 0:
            save_image(denorm(real_cpu.cpu()).float(), '../CW_DCGAN/real_samples.png')

        fake = model_G(fixed_noise)
        save_image(denorm(fake.cpu()).float(), '../CW_DCGAN/fake_samples_epoch_%03d.png' % epoch)
        train_losses_D.append(train_loss_D / len(loader_train))
        train_losses_G.append(train_loss_G / len(loader_train))

        import imageio

        images = []
        for epoch in range(num_epochs):
            images.append(imageio.imread('../CW_DCGAN/fake_samples_epoch_%03d.png' % epoch))
        imageio.mimsave('/GAN_movie.gif', images)

        # save losses and models
        torch.save(model_G.state_dict(), '../CW_DCGAN/DCGAN_model_G.pth')
        torch.save(model_D.state_dict(), '../CW_DCGAN/DCGAN_model_D.pth')

        it = iter(loader_test)
        sample_inputs, _ = next(it)
        fixed_input = sample_inputs[0:32, :, :, :]

        # visualize the original images of the last batch of the test set
        img = make_grid(denorm(fixed_input), nrow=4, padding=2, normalize=False,
                        range=None, scale_each=False, pad_value=0)

        fig = plt.figure()
        fig.set_size_inches(8, 8)
        show(img)

        # load the model
        model_G.load_state_dict(torch.load('../CW_DCGAN/DCGAN_model_G.pth'))
        input_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)

        with torch.no_grad():
            fig = plt.figure()
        fig.set_size_inches(12, 24)
        # visualize the generated images
        generated = model_G(input_noise).cpu()
        generated = make_grid(denorm(generated)[:32], nrow=8, padding=2, normalize=False,
                              range=None, scale_each=False, pad_value=0)
        show(generated)

        import matplotlib.pyplot as plt

        fig = plt.figure()
        fig.set_size_inches(12, 9)
        plt.plot(list(range(0, np.array(train_losses_D).shape[0])), np.array(train_losses_D), label='loss_D')
        plt.plot(list(range(0, np.array(train_losses_G).shape[0])), np.array(train_losses_G), label='loss_G')
        plt.legend()
        plt.title('Train Losses')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.show()
