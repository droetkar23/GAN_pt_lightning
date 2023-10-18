from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import pytorch_lightning as pl
from torch import nn, optim
import torch
import torchvision

from modules import Discriminator, Generator

# fixed_noise = torch.randn((batch_size, z_dim)).to(device)

class DCGanCelebA(pl.LightningModule):
    def __init__(self, z_dim, channels_image, features_g, features_d, learning_rate) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.z_dim = z_dim
        self.gen = Generator(z_dim=z_dim, channels_image=channels_image, features_g=features_g)
        self.disc = Discriminator(channels_image=channels_image, features_d=features_d)
        self.criterion = nn.BCELoss()

        self.automatic_optimization = False

        # need a fixed sample noise for logging
        self.fixed_noise = torch.randn(32, z_dim, 1, 1)

        # defining self.example_input_array will add the
        # computational graph to the logger automatically
        self.example_input_array = torch.zeros(1, z_dim, 1, 1)
        # print(f"{self.device=}")
        # print(f"{self.disc.device=}")
        # print(f"{self.gen.device=}")
        

    def custom_weight_init(self):
    # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def test(self):
        N, in_channels, H, W = 8, self.hparams.channels_image, 64, 64
        noise_dim = self.hparams.z_dim
        x = torch.randn((N, in_channels, H, W), device=self.device)
        print(f"{self.device=}")
        print(f"{x.device=}")
        # disc = self.disc(in_channels, 8)
        assert self.disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
        # gen = self.gen(noise_dim, in_channels, 8)
        z = torch.randn((N, noise_dim, 1, 1), device=self.device)
        assert self.gen(z).shape == (N, in_channels, H, W), "Generator test failed"
        print("Success, tests passed!")



    def forward(self, x) -> Any:
        return self.gen(x)


    def training_step(self, batch) -> STEP_OUTPUT:
        # implement training step
        real_img, _ = batch
        # real_img = real_img.view(-1, 784)

        opt_gen, opt_disc = self.optimizers()

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        #generate noise for generator
        noise = torch.randn((real_img.shape[0], self.z_dim, 1, 1), device=self.device)
        #generate images from it
        fake_img = self.gen(noise)

        # discriminator loss from real and fake image
        disc_real = self.disc(real_img).view(-1)
        lossD_real = self.criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = self.disc(fake_img).view(-1)
        lossD_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        self.log("lossD", lossD, prog_bar=True)

        opt_disc.zero_grad()
        self.manual_backward(lossD, retain_graph=True)   # keep 
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = self.disc(fake_img).view(-1)
        lossG = self.criterion(output, torch.ones_like(output))
        self.log("lossG", lossG, prog_bar=True)
        
        opt_gen.zero_grad()
        self.manual_backward(lossG)
        opt_gen.step()

    def on_validation_epoch_end(self):
        # log sampled images
        sample_imgs = self(self.fixed_noise)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        grid = torchvision.utils.make_grid(self.real_image_sample)
        self.logger.experiment.add_image("real_images", grid, self.current_epoch)
        

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        real_img, _ = batch
        if batch_idx==0:
            self.real_image_sample = real_img[:self.fixed_noise.shape[0]]
        pass
        
    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        # implement test step
        pass

    def configure_optimizers(self) -> Any:
        opt_g = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.disc.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return opt_g, opt_d
    