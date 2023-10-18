import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split



# # custom dataset
# class CustomDataset():
#     # init

#     # len

#     # getitem



class plDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self) -> None:
        # eg download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage: str) -> None:
        # setup the dataset, train/val split, test dataset, transforms
        train_val_ds = None
        self.train_ds, self.val_ds = random_split(train_val_ds, [50000, 10000])
        self.test_ds = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
