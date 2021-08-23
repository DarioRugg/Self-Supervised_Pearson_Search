import os
from os.path import join

import torch

from models.model_pretrained import Model as PretrainedNet
from dataloaders.lightning_dataloader import SelfSupervisionDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))

    data_path = join("..", "..", "data", "CUHK-SYSU", "Image", "SSM")
    checkpoint_path = join("checkpoints")
    dataset = SelfSupervisionDataModule(data_path)

    # Network initialize
    model = PretrainedNet(lr=cfg.model.lr)

    # Train the Model
    logger = TensorBoardLogger("tb_logs", name=cfg.sim_name)
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=cfg.model.epochs, logger=logger)

    trainer.fit(model, datamodule=dataset)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    model.save(join(checkpoint_path, cfg.sim_name+"_checkpoint"))


if __name__ == "__main__":
    main()
