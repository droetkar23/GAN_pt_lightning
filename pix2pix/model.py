from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import pytorch_lightning as pl
from torch import nn, optim
import torch
import torchvision

from modules import Discriminator, Generator

# fixed_noise = torch.randn((batch_size, z_dim)).to(device)

class Pix2PixShoes(pl.LightningModule):
    def __init__(self, lr=1e-5, l1_lambda=100) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.gen = Generator()
        self.disc = Discriminator()
        self.criterionBCE = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()

        self.automatic_optimization = False

        # defining self.example_input_array will add the
        # computational graph to the logger automatically
        self.example_input_array = torch.zeros(1, 3, 256, 256)

    def forward(self, x) -> Any:
        return self.gen(x)

    def configure_optimizers(self) -> Any:
        opt_g = optim.Adam(self.gen.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.disc.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return opt_g, opt_d

    def custom_weight_init(self):
    # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)


    def training_step(self, batch) -> STEP_OUTPUT:
        # implement training step
        image_A, image_B = batch

        opt_gen, opt_disc = self.optimizers()

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # Generator forward pass
        image_A_to_B = self.gen(image_A)

        # discriminator loss from original and translated image
        disc_A = self.disc(torch.cat((image_A, image_B), dim=1))     #.view(-1)
        lossD_A = self.criterionBCE(disc_A, torch.ones_like(disc_A))

        disc_A_to_B = self.disc(torch.cat((image_A, image_A_to_B), dim=1))   #.view(-1)
        lossD_A_to_B = self.criterionBCE(disc_A_to_B, torch.zeros_like(disc_A_to_B))

        lossD = (lossD_A + lossD_A_to_B) / 2
        self.log("lossD", lossD, prog_bar=True)

        opt_disc.zero_grad()
        self.manual_backward(lossD, retain_graph=True)   # keep 
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        # cat on dim=1 (color channels of pictures)
        output = self.disc(torch.cat((image_A, image_A_to_B), dim=1))    #.view(-1)
        # for pix2pix a sum of BCE and L1 loss is used
        lossG_BCE = self.criterionBCE(output, torch.ones_like(output))
        lossG_L1 = self.criterionL1(image_A_to_B, image_B)
        lossG = lossG_BCE + self.hparams.l1_lambda * lossG_L1

        self.log("lossG", lossG, prog_bar=True)
        
        opt_gen.zero_grad()
        self.manual_backward(lossG)
        opt_gen.step()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        image_A, image_B = batch
        if batch_idx==0:
            batch_size = image_A.shape[0]
            num_samples = min(8, batch_size)
            self.image_A_B_samples = image_A[:num_samples], image_B[:num_samples]

    def on_validation_epoch_end(self):
        # log sampled images
        image_A_samples, image_B_samples = self.image_A_B_samples
        image_A_to_B_samples = self(image_A_samples)
        # dim=2 corresponds to the x-axis of the images, so the comparison is side to side TODO which dim to cat on???
        comparison_samples = torch.cat((image_A_samples, image_B_samples, image_A_to_B_samples), dim=2)
        # print(f"{comparison_samples[0].shape=}")
        # print(f"{comparison_samples[0].dtype=}")
        # print(f"{comparison_samples[0]=}")
        grid = torchvision.utils.make_grid(comparison_samples, nrow=8)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        
        

    
    # def test(self):
    #     N, in_channels, H, W = 8, self.hparams.channels_image, 64, 64
    #     noise_dim = self.hparams.z_dim
    #     x = torch.randn((N, in_channels, H, W), device=self.device)
    #     print(f"{self.device=}")
    #     print(f"{x.device=}")
    #     # disc = self.disc(in_channels, 8)
    #     assert self.disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    #     # gen = self.gen(noise_dim, in_channels, 8)
    #     z = torch.randn((N, noise_dim, 1, 1), device=self.device)
    #     assert self.gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    #     print("Success, tests passed!")