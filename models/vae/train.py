if __name__ == "__main__":
    import os
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import imageio
    import os
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from models.vae.vae import ConvVAE

    from utils.data_helpers import denorm_for_sigmoid
    from utils.model_helpers import loss_function_VAE
    import imageio

    import sys

    print(sys.path)

    from utils.model_helpers import get_device

    # device selection
    device = get_device()
    print(device)

    torch.manual_seed(0)

    if not os.path.exists('../../CW_VAE/MNIST'):
        os.makedirs('../../CW_VAE/MNIST')

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
        "../../data/", train=True, download=True, transform=transform
    )
    test_dat = datasets.MNIST("../../data/", train=False, transform=transform)

    loader_train = DataLoader(train_dat, batch_size, shuffle=True)
    loader_test = DataLoader(test_dat, batch_size, shuffle=False)

    sample_inputs, _ = next(iter(loader_test))
    fixed_input = sample_inputs[:32, :, :, :]

    save_image(fixed_input, '../../CW_VAE/MNIST/image_original.png')

    model = ConvVAE(latent_dim, amp=1).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    print(model)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_function = loss_function_VAE

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
            loss, BCE, KLD = loss_function(recon_batch, img, mu, logvar, beta)
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
        save_image(recon.float(), '../../CW_VAE/MNIST/reconstructed_epoch_{}.png'.format(epoch))
        print(f"Train: Loss: {train_loss}, "
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
        print(f"Test: Total Loss: {test_loss}, "
              f"Reconstruction Loss: {test_BCE / len(loader_test)} and "
              f"KLD: {test_KLD / len(loader_test)}")

    images = []
    for epoch in range(num_epochs):
        images.append(imageio.imread('../../CW_VAE/MNIST/reconstructed_epoch_{}.png'.format(epoch)))
    imageio.mimsave('../../CW_VAE/VAE_movie.gif', images)

    # save the model
    torch.save(model.state_dict(), '../../CW_VAE/MNIST/VAE_model.pth')
