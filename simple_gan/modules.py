from torch import nn


# optionally create building blocks for the model here and import them in model.py

class Discriminator(nn.Sequential):
    def __init__(self, img_dim):
        super().__init__(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

class Generator(nn.Sequential):
    def __init__(self, z_dim, img_dim):
        super().__init__(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )