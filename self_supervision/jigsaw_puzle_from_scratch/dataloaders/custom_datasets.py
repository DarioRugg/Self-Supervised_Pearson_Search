from os.path import join
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image
from os import listdir

class CustomDataset(Dataset):
    def __init__(self, data_path, classes=1000):
        super().__init__()

        self.data_path = data_path

        # names of the images
        self.names = listdir(self.data_path)
        self.N = len(self.names)
        self.permutations = self.__retrive_permutations(classes)

        self.__image_transformer = transforms.Compose([
            transforms.RandomGrayscale(p=0.3),
            transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(saturation=0.01),
            transforms.ToTensor()])

    def __getitem__(self, index):
        framename = join(self.data_path, self.names[index])

        img = Image.open(framename).convert('RGB')

        img = self.__image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9

        tiles_limits = lambda dim_size: zip(np.linspace(0,dim_size, 4, dtype=int)[:-1],
                                            np.linspace(0,dim_size, 4, dtype=int)[1:])

        for n, (x_limits, y_limits) in list(enumerate(product(tiles_limits(img.size[0]), tiles_limits(img.size[1])))):
            tile = img.crop(x_limits + y_limits)
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)

        return data, int(order), tiles

    def __len__(self):
        return len(self.names)
