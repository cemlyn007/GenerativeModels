if __name__ == "__main__":
    import os
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import imageio
    from torchvision.utils import save_image
    from models.vae import vae
    from utils.data_helpers import denorm_for_sigmoid
    from utils.model_helpers import loss_function_VAE, get_device
    import argparse


    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        parser.add_argument("--beta", type=float, default=1.)
        parser.add_argument("--latent_dim", type=int, default=16)
        parser.add_argument("--n_epochs", type=int, default=20)
        parser.add_argument("--NUM_TRAIN", type=int, default=49000)

        parser.add_argument("--data_dir", type=str, default='data')
        parser.add_argument("--save_dir", type=str, default='MNIST')

        return parser.parse_args()


    args = get_args()

    # device selection
    device = get_device()
    print(device)

    torch.manual_seed(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(os.path.join(args.save_dir, 'images')):
        os.makedirs(os.path.join(args.save_dir, 'images'))

    # Modify this line if you need to do any input transformations (optional).
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    denorm = denorm_for_sigmoid

    """## Data loading"""

    train_dat = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    test_dat = datasets.MNIST(args.data_dir, train=False, transform=transform)

    loader_train = DataLoader(train_dat, args.batch_size, shuffle=True)
    loader_test = DataLoader(test_dat, args.batch_size, shuffle=False)

    sample_inputs, _ = next(iter(loader_test))
    fixed_input = sample_inputs[:32, :, :, :]

    save_image(fixed_input, os.path.join(args.save_dir, 'images', 'image_original.png'))

    model = vae.ConvVAE(args.latent_dim).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    print(model)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_function = loss_function_VAE

    epoch_loss = []
    reconstruction_list = []
    KLD_list = []
    test_loss_lst = []
    test_BCE_lst = []
    test_KLD_lst = []

    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0
        reconstruction_loss = 0
        KLD_loss = 0

        for batch_idx, data in enumerate(loader_train):
            optimizer.zero_grad()
            img, _ = data

            img = img.to(device)
            optimizer.zero_grad()
            # forward
            recon_batch, mu, logvar = model(img)
            loss, BCE, KLD = loss_function(recon_batch, img, mu, logvar, args.beta)
            # backward
            loss.backward()
            train_loss += loss.item()
            reconstruction_loss += BCE.item()
            KLD_loss += KLD.item()
            # print(f"Debug Train: Avg Loss: {loss.item()}, Reconstruction Loss: {BCE.item()} and KLD: {KLD.item()}")
            optimizer.step()
        # print out losses and save reconstructions for every epoch
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, args.n_epochs, train_loss / len(loader_train)))
        recon = denorm(model(fixed_input.to(device))[0])

        save_image(recon.float(), os.path.join(args.save_dir, 'images', f'reconstructed_epoch_{epoch}.png'))
        print(f"Train: Loss: {train_loss / len(loader_train)}, "
              f"Reconstruction Loss: {reconstruction_loss / len(loader_train)} and "
              f"KLD Loss: {KLD_loss / len(loader_train)}")
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
                loss, BCE, KLD = loss_function(recon_batch, img, mu, logvar, args.beta)
                test_loss += loss
                test_BCE += BCE
                test_KLD += KLD
            # reconstruct and save the last batch
            recon_batch = model(recon_batch.to(device))
            img = denorm(img.cpu())

        test_loss_lst.append(test_loss)
        test_BCE_lst.append(test_BCE)
        test_KLD_lst.append(test_KLD)
        print(f"Test: Total Loss: {test_loss / len(loader_test)}, "
              f"Reconstruction Loss: {test_BCE / len(loader_test)} and "
              f"KLD: {test_KLD / len(loader_test)}")

    images = []
    for epoch in range(args.n_epochs):
        images.append(imageio.imread(os.path.join(args.save_dir, 'images', f'reconstructed_epoch_{epoch}.png')))
    imageio.mimsave(os.path.join(args.save_dir, 'VAE_movie.gif'), images)

    # save the model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'VAE_model.pth'))
