import torch


def get_device(use_gpu=True, device_idx=0):
    if use_gpu:
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def loss_function_VAE(recon_x, x, mu, logvar, beta):
    bce = torch.sum(torch.nn.functional.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784),
                                                             reduction='none'), dim=1)
    expected_bce = torch.mean(bce)
    batch_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    loss = expected_bce + beta * batch_kld
    return loss, expected_bce, batch_kld
