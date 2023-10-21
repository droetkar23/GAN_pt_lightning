# GAN_pt_lightning
Using the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) PyTorch framework to implement the training of Generative Adverserial Networks (GANs) following a series of youtube tutorials [Generative Adversarial Networks (GANs) Playlist](https://www.youtube.com/playlist?list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va).


## Setup

The required packages are listed in [environment.yml](environment.yml).
If you are using a conda like package manager, e.g. [miniforge](https://github.com/conda-forge/miniforge#install), you can create a virtual environment with 
```
mamba create --file environment.yml
```
On Windows use the Miniforge Prompt or initialize your prompt of choice, eg. powershell with
(see https://docs.conda.io/projects/conda/en/latest/commands/init.html).
```
conda init powershell
```

In addition to the packages required to run the training you will want to install tensorboard as a logging interface.
```
mamba install tensorboard tensorboardX
```


## Usage

The [lightning_template](lightning_template/) folder contains skeleton files with the lightning modules and their methods that need to be implemented. This includes a LightningModule in [model.py](lightning_template/model.py)  and a LightningDataModule in [dataset.py](lightning_template/dataset.py). [config.yml](lightning_template/config.yml) contains parameters that are passed as keyword arguments to the model, dataset and trainer objects in [train.py](lightning_template/train.py).

After implementing a model and dataset start the training with
```
python train.py
```
Monitor progress with a tensorboard dashboard through a web browser
```
tensorboard --log-dir path/to/logs
```
following the provided link (eg. localhost:6006)

