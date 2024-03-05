import pytorch_lightning as pl
from model import Pix2PixShoes     #change to name in model.py
from dataset import plDataModule
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
import torch



if __name__ == "__main__":

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    model_parameters = config["model_parameters"]
    dataset_parameters = config["dataset_parameters"]
    trainer_parameters = config["trainer_parameters"]

    logger = TensorBoardLogger(save_dir="tb_logs", name="pix2pix", log_graph=True)
    trainer = pl.Trainer(
        logger=logger,
        # profiler="simple",
        **trainer_parameters
    )

    with trainer.init_module():
        # lightning initializes model on cpu but
        # we have some tensors in init that we want on the gpu
        model = Pix2PixShoes(**model_parameters)

    # N, in_channels, H, W = 8, model.hparams.channels_image, 64, 64
    # noise_dim = model.hparams.z_dim
    # x = torch.randn((N, in_channels, H, W), device="cuda")
    # # print(f"{x.device=}")
    # # print(f"{torch.current_device()}")
    # # quit()
    # # disc = Discriminator(in_channels, 8)
    # assert model.disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    # # gen = Generator(noise_dim, in_channels, 8)
    # z = torch.randn((N, noise_dim, 1, 1), device="cuda")
    # assert model.gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    # print("Success, tests passed!")
    # # model.test()


    dm = plDataModule(**dataset_parameters)
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model,dm)

