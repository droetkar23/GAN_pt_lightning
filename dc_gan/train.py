import pytorch_lightning as pl
from model import DCGanCelebA     #change to name in model.py
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

    logger = TensorBoardLogger("tb_logs", name="dc_gan_celeba", log_graph=True)
    trainer = pl.Trainer(
        logger=logger,
        # profiler="simple",
        **trainer_parameters
    )
    with trainer.init_module():
        # lightning initializes model on cpu but
        # we have some tensors in init that we want on the gpu
        model = DCGanCelebA(**model_parameters)

    model.custom_weight_init()

    dm = plDataModule(**dataset_parameters)
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model,dm)

