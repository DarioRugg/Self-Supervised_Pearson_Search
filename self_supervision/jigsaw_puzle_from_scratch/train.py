from os.path import join

import torch

from models.model_pretrained import Model as PretrainedNet
from dataloaders.lightning_dataloader import SelfSupervisionDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


args = {
    "lr": 1e-4

}

def main():
    data_path = join("..", "..", "data", "CUHK-SYSU", "Image", "SSM")
    dataset = SelfSupervisionDataModule(data_path)

    # Network initialize
    model = PretrainedNet()

    # Train the Model
    logger = TensorBoardLogger("tb_logs", name="test")
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0, logger=logger)

    trainer.fit(model, datamodule=dataset)

    model.save(join("checkpoints", "test_checkpoint"))


if __name__ == "__main__":
    main()
