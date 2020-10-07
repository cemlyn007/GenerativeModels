import torch.nn as nn

from utils.module_helpers import PrintTensors


class CTGenerator(nn.Module):
    def __init__(self, latent_vector_size: int):
        super(CTGenerator, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_vector_size, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, z):
        return self.decode(z)


# # NNResizeConvGenerator
# class NNResizeConvGenerator(nn.Module):
#     def __init__(self, latent_vector_size: int):
#         super(NNResizeConvGenerator, self).__init__()
#         mode = "bilinear"
#
#         self.decoder = nn.Sequential(
#             PrintTensors("Input"),
#             nn.ConvTranspose2d(latent_vector_size, 1024, kernel_size=3),
#             # PrintTensors("1"),
#             # nn.Upsample(size=3, mode=mode),
#             nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=2),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             PrintTensors("1"),
#             nn.Upsample(size=4, mode=mode),
#             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             PrintTensors("2"),
#             nn.Upsample(size=6, mode=mode),
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             PrintTensors("3"),
#             nn.Upsample(size=10, mode=mode),
#             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             PrintTensors("4"),
#             nn.Upsample(size=18, mode=mode),
#             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             PrintTensors("5"),
#             nn.Upsample(size=34, mode=mode),
#             nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3),
#             nn.Tanh(),
#             PrintTensors("Output"),
#         )
#
#     def decode(self, z):
#         x = self.decoder(z)
#         return x
#
#     def forward(self, z):
#         return self.decode(z)
