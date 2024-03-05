import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision import datasets, transforms, io
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import glob
from pathlib import Path
import requests
from tqdm.auto import tqdm
import shutil
import tarfile


# custom dataset
class ShoeDataset(Dataset):
    def __init__(self,
                 root_dir="datasets/edges2shoes",
                 image_dir="images/edges2shoes/train",
                 download=False, extract=True,
                 delete_after_extract=False,
                 transform=None,
                 ) -> None:
        self.root_dir = Path(root_dir)
        self.download_url = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz"
        self.archive_download_path = self.root_dir / Path(self.download_url.rpartition("/")[-1])
        self.image_dir = self.root_dir / image_dir
        self.transform = transform

        if not self.root_dir.exists():
            Path.mkdir(self.root_dir, parents=True)

        if download and self.archive_download_path.exists()==False:
            # make an HTTP request within a context manager
            with requests.get(self.download_url, stream=True) as r:
                # check header to get content length, in bytes
                total_length = int(r.headers.get("Content-Length"))              
                # implement progress bar via tqdm
                with tqdm.wrapattr(r.raw, "read", total=total_length, desc="")as raw:              
                    # save the output to a file
                    with open(self.archive_download_path, 'wb')as output:
                        shutil.copyfileobj(raw, output)
        
        if extract and self.archive_download_path.exists():
            with tarfile.open(self.archive_download_path, "r") as tar:
                tar.extractall(self.root_dir)
            if delete_after_extract:
                Path.unlink(self.archive_download_path)

        self.image_files_list = list(self.image_dir.iterdir())

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        image_pair = io.read_image(str(self.image_files_list[idx]))
        image_A = image_pair[:,:,:256]
        image_B = image_pair[:,:,256:]
        if self.transform is not None:
            image_A, image_B = self.transform(image_A), self.transform(image_B)
        return image_A, image_B

    # def __getitems__(self):
    #     ...



class plDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=0, train_ds_size=1.0, val_ds_size=1.0) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ds_size = train_ds_size
        self.val_ds_size = val_ds_size

        self.transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                # transforms.Normalize(
                #     [0.5 for _ in range(channels_image)], [0.5 for _ in range(channels_image)]
                # ),
            ]
        )
    
    def prepare_data(self) -> None:
        ShoeDataset(root_dir="datasets/edges2shoes", download=True, extract=False)

    def setup(self, stage: str) -> None:

        self.train_ds = random_split(
            ShoeDataset(
            image_dir="images/edges2shoes/train",
            extract=False,
            transform=self.transforms,
            ),
            [self.train_ds_size, 1-self.train_ds_size]
        )[0]

        self.val_ds = random_split(
            ShoeDataset(
            image_dir="images/edges2shoes/val",
            extract=False,
            transform=self.transforms,
            ),
            [self.val_ds_size, 1-self.val_ds_size]
        )[0]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )
    
    # def test_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(
    #         self.test_ds,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         persistent_workers=True,
    #         shuffle=False
    #     )
