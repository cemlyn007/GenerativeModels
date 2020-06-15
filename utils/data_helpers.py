import numpy as np
import torch


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
