import pytorch_lightning as pl
from model import SimpleMnistGan     #change to name in model.py
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

    # logger = TensorBoardLogger("tb_logs", name="simple_mnist_gan_v0")
    trainer = pl.Trainer(
        # logger=logger,
        # profiler="simple",
        **trainer_parameters
    )
    with trainer.init_module():
        # lightning initializes model on cpu but
        # we have some tensors in init that we want on the gpu
        model = SimpleMnistGan(**model_parameters)

    dm = plDataModule(**dataset_parameters)
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model,dm)

