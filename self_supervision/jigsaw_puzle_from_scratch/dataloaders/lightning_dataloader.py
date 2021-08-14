from torch.utils.data import DataLoader
from self_supervision.jigsaw_puzle_from_scratch.dataloaders.custom_datasets import CustomDataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", permutations=1000, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_split, self.val_split, self.test_split = CustomDataLoader(data_dir, permutations)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
