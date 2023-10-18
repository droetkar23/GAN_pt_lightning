from torch import nn


# optionally create building blocks for the model here and import them in model.py

class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, use_bn=True):
        modules = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        ]
        if not use_bn:
            del modules[1]
        super().__init__(*modules)


class ConvTransposeBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, use_bn=True):
        modules = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        ]
        if not use_bn:
            del modules[1]
        super().__init__(*modules)


class Discriminator(nn.Sequential):
    def __init__(self, channels_image, features_d, kernel_size=4, stride=2, padding=1):
        super().__init__(
            # input: (N x channels_img x 64 x 64)
            ConvBlock(channels_image, features_d, kernel_size, stride, padding, use_bn=False),  # 32x32
            ConvBlock(features_d, features_d*2, kernel_size, stride, padding),                  # 16x16
            ConvBlock(features_d*2, features_d*4, kernel_size, stride, padding),                # 8x8
            ConvBlock(features_d*4, features_d*8, kernel_size, stride, padding),                # 4x4
            nn.Conv2d(features_d*8, 1, kernel_size, stride, padding=0),                           # 1x1
            nn.Sigmoid()
        )


class Generator(nn.Sequential):
    def __init__(self, z_dim, channels_image, features_g, kernel_size=4, stride=2, padding=1):
        super().__init__(
            # input: (N x z_dim x 1 x 1)
            ConvTransposeBlock(z_dim, features_g*16, kernel_size, stride=1, padding=0),
            ConvTransposeBlock(features_g*16, features_g*8, kernel_size, stride, padding),
            ConvTransposeBlock(features_g*8, features_g*4, kernel_size, stride, padding),
            ConvTransposeBlock(features_g*4, features_g*2, kernel_size, stride, padding),
            nn.ConvTranspose2d(features_g*2, channels_image, kernel_size, stride, padding),
            nn.Tanh()
        )