import os
from os.path import join
import numpy as np
from time import time

import torch
import torch.nn as nn

from models.model_pretrained import Model as PretrainedNet
# from self_supervision.jigsaw_puzle_from_scratch.models.model_pretrained import Model as PretrainedNet
from dataloaders.lightning_dataloader import SelfSupervisionDataModule
# from self_supervision.jigsaw_puzle_from_scratch.dataloaders.lightning_dataloader import SelfSupervisionDataModule
from pytorch_lightning import Trainer


def main():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = join("data", "CUHK-SYSU", "Image", "SSM")
    # data_path = join("..", "..", "data", "CUHK-SYSU", "Image", "SSM")
    dataset = SelfSupervisionDataModule(data_path)

    # Network initialize
    model = PretrainedNet()

    # Train the Model
    trainer = Trainer()

    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    main()
