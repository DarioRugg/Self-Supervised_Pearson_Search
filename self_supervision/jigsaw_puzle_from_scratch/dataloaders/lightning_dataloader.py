from torch.utils.data import DataLoader, random_split
from dataloaders.custom_datasets import CustomDataset
# from self_supervision.jigsaw_puzle_from_scratch.dataloaders.custom_datasets import CustomDataset
import torch
import pytorch_lightning as pl


class SelfSupervisionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = None, permutations=1000, batch_size: int = 16, num_workers: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        entire_dataset = CustomDataset(data_dir, permutations)
        n = len(entire_dataset)

        self.train_split, self.val_split, self.test_split = \
            random_split(entire_dataset, [round(n * 0.7), round(n * 0.15), n-round(n * 0.7)-round(n * 0.15)],
                         generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_split, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, num_workers=self.num_workers)
