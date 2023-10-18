from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import pytorch_lightning as pl
from torch import nn, optim

class pl_module(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # define used modules

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # implement the forward pass
        pass

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # implement training step 
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        # implement validation step
        pass
        
    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        # implement test step
        pass

    def configure_optimizers(self) -> Any:
        # example optimizer
        return optim.Adam(self.parameters(), lr=0.001)