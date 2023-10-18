from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import pytorch_lightning as pl
from torch import nn, optim
import torch
import torchvision

from modules import Discriminator, Generator

# fixed_noise = torch.randn((batch_size, z_dim)).to(device)

class SimpleMnistGan(pl.LightningModule):
    def __init__(self, z_dim, img_dim, learning_rate) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.z_dim = z_dim
        self.img_dim = img_dim
        self.disc = Discriminator(img_dim=img_dim)
        self.gen = Generator(z_dim=z_dim, img_dim=img_dim)
        self.criterion = nn.BCELoss()

        self.automatic_optimization = False

        # need a fixed sample noise for logging
        self.fixed_noise = torch.randn(32, z_dim)

        # defining self.example_input_array will add the
        # computational graph to the logger automatically
        self.example_input_array = torch.zeros(1, z_dim)


    def forward(self, x) -> Any:
        return self.gen(x)


    def training_step(self, batch) -> STEP_OUTPUT:
        # implement training step
        real_img, _ = batch
        real_img = real_img.view(-1, 784)

        opt_gen, opt_disc = self.optimizers()

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        #generate noise for generator
        noise = torch.randn((real_img.shape[0], self.z_dim))
        noise = noise.type_as(real_img)
        #generate images from it
        fake = self.gen(noise)
        # log sampled images
        # sample_fakes = fake[:32]
        # grid = torchvision.utils.make_grid(sample_fakes)
        # self.logger.experiment.add_image("generated_images", grid, 0)
        # discriminator loss from real and fake image
        disc_real = self.disc(real_img).view(-1)
        lossD_real = self.criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = self.disc(fake)
        lossD_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        self.log("lossD", lossD, prog_bar=True)
        self.disc.zero_grad()
        lossD.backward(retain_graph=True)   # keep 
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = self.disc(fake).view(-1)
        lossG = self.criterion(output, torch.ones_like(output))
        self.log("lossG", lossG, prog_bar=True)
        self.gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    def on_validation_epoch_end(self):
        # z = self.fixed_noise.type_as(self.example_input_array)
        # print(z.device)

        # log sampled images
        sample_imgs = self(self.fixed_noise).view(-1,1,28,28)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        # implement validation step
        pass
        
    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        # implement test step
        pass

    def configure_optimizers(self) -> Any:
        opt_g = optim.Adam(self.gen.parameters(), lr=self.lr)
        opt_d = optim.Adam(self.disc.parameters(), lr=self.lr)
        return opt_g, opt_d
    