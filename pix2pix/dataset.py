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
    def __init__(self, data_dir, batch_size, split=[0.8, 0.2, 0.0], num_workers=0, resize_to=64, channels_image=1) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_to),
                transforms.CenterCrop(resize_to),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(channels_image)], [0.5 for _ in range(channels_image)]
                ),
            ]
        )
    
    def prepare_data(self) -> None:
        # eg download
        # datasets.MNIST(self.data_dir, train=True, download=True)
        # datasets.MNIST(self.data_dir, train=False, download=True)
        datasets.CelebA(root=self.data_dir, download=True)
        
    def setup(self, stage: str) -> None:
        # setup the dataset, train/val split, test dataset, transforms
        # train_val_ds = datasets.MNIST(
        #     self.data_dir,
        #     train=True,
        #     download=False,
        #     transform=self.transforms
        # )
        
        # self.train_ds, self.val_ds = random_split(train_val_ds, [50000, 10000])

        # self.test_ds = datasets.MNIST(
        #     self.data_dir,
        #     train=False,
        #     download=False,
        #     transform=self.transforms
        # )

        ds = datasets.CelebA(root=self.data_dir, transform=self.transforms)
        self.train_ds, self.val_ds, self.test_ds = random_split(ds, self.split)

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
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )
