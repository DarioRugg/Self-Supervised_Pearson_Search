from os.path import join

import torch

from models.model_pretrained import Model as PretrainedNet
from dataloaders.lightning_dataloader import SelfSupervisionDataModule
from pytorch_lightning import Trainer


def main():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = join("..", "..", "data", "CUHK-SYSU", "Image", "SSM")
    dataset = SelfSupervisionDataModule(data_path)

    # Network initialize
    model = PretrainedNet()

    # Train the Model
    trainer = Trainer(gpus=device.index)

    trainer.fit(model, datamodule=dataset)

    model.save(join("checkpoints", "test_checkpoint"))


if __name__ == "__main__":
    main()
