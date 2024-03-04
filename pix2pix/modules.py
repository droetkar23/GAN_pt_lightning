from torch import nn
import torch


# optionally create building blocks for the model here and import them in model.py

class ConcatBlock(nn.Module):
    def __init__(self, concat_dim=1) -> None:
        super().__init__()
        self.concat_dim = concat_dim
    def forward(self, *tensors):
        return torch.cat(tensors, self.concat_dim)

class ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 use_bn=True,
                 activation=nn.LeakyReLU(0.2),
                 use_dropout=False,
                 ):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode="reflect")
            ]
        if use_bn:
            modules.append(nn.BatchNorm2d(out_channels))
        if activation:
            modules.append(activation)
        if use_dropout:
            modules.append(nn.Dropout(0.5))
        super().__init__(*modules)


class ConvTransposeBlock(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 use_bn=True,
                 activation=nn.ReLU(),
                 use_dropout=False,
                 ):
        self.out_channels=out_channels
        modules = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            ]
        if use_bn:
            modules.append(nn.BatchNorm2d(out_channels))
        if activation:
            modules.append(activation)
        if use_dropout:
            modules.append(nn.Dropout(0.5))
        super().__init__(*modules)


class Discriminator(nn.Sequential):
    def __init__(self, in_channels=3, features=(64, 128, 256, 512), **kwargs):
        super().__init__(
            # input: (N x in_channels x 64 x 64)
            # needs concatenated input (original and desired/generated image)
            ConvBlock(in_channels*2, features[0], use_bn=False),
            ConvBlock(features[0], features[1]),
            ConvBlock(features[1], features[2]),
            ConvBlock(features[2], features[3], stride=1),
            ConvBlock(features[3], 1, stride=1, use_bn=False, activation=None),
        )

    def test(self):
        x = torch.randn((1,3,256,256))
        y = torch.randn((1,3,256,256))
        pred = self(torch.cat((x,y), dim=1))
        print(f"{pred.shape=}")


class Generator(nn.Module):

    def __init__(self,
                 img_channels=3,
                 feature_channels=(64,128,256,512,512,512,512),
                 decoder_dropout_idcs=(0,1),
                 **kwargs
                 ) -> None:
        super().__init__()
        # making this list is ugly
        self.input_conv = ConvBlock(img_channels, feature_channels[0])

        down_modules = [
            ConvBlock(in_ch, out_ch) for in_ch, out_ch in
            zip(feature_channels[:-1], feature_channels[1:])
            ]
        self.down_modules = nn.ModuleList(down_modules)

        self.bottle_neck = nn.ModuleList([
            ConvBlock(feature_channels[-1], feature_channels[-1], use_bn=False, activation=nn.ReLU()),
            ConvTransposeBlock(feature_channels[-1], feature_channels[-1], use_dropout=True)
        ])

        up_modules = [
            ConvTransposeBlock(in_ch*2, out_ch, use_dropout=True) if i in decoder_dropout_idcs else
            ConvTransposeBlock(in_ch*2, out_ch, use_dropout=False) for i, (in_ch, out_ch) in
            enumerate(zip(feature_channels[::-1], feature_channels[-2::-1]))
            ]
        self.up_modules = nn.ModuleList(up_modules)

        self.output_conv = ConvTransposeBlock(feature_channels[0], img_channels, activation=nn.Tanh())


    def forward(self,x):

        x = self.input_conv(x)

        down_outputs = []
        for dm in self.down_modules:
            x = dm(x)
            down_outputs.append(x)

        for bn in self.bottle_neck:
            x = bn(x)

        for do, um in zip(down_outputs[::-1], self.up_modules):
            x = um(torch.cat((x, do), dim=1))

        x = self.output_conv(x)

        return x


    def test(self):
        self.train(False)
        x = torch.randn((1,3,256,256))
        pred = self(x)
        print(f"{pred.shape=}")
