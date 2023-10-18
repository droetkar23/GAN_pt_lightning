import pytorch_lightning as pl
from model import pl_module     #change to name in model.py
from dataset import plDataModule
import yaml



if __name__ == "__main__":

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    hyperparameters = config["hyperparameters"]
    dataset_parameters = config["dataset_parameters"]
    compute_parameters = config["compute_parameters"]

    model = pl_module()
    dm = plDataModule(data_dir="datset/", batch_size=1, num_workers=3)
    trainer = pl.Trainer(
        accelerator=compute_parameters["accelerator"],
        devices=compute_parameters["devices"],
        min_epochs=1,
        max_epochs=compute_parameters["num_epochs"])
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model,dm)

