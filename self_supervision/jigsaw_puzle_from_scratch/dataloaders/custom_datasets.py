from os.path import join
from os import listdir
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

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

    def __retrive_permutations(self, classes):
        all_perm = np.load(join("permutations", f'permutations_{classes}.npy'))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def __getitem__(self, index):
        img = Image.open(join(self.data_path, self.names[index])).convert('RGB')

        img = self.__image_transformer(img)

        tiles_limits = lambda dim_size: zip(np.linspace(0,dim_size, 4, dtype=int)[:-1],
                                            np.linspace(0,dim_size, 4, dtype=int)[1:])

        tiles = []

        for x_limits, y_limits in list(product(tiles_limits(img.size[0]), tiles_limits(img.size[1]))):
            tile = img.crop(x_limits + y_limits)
            print(f" --> tiles info;\n  shape: {tile.shape}\n  limits: {x_limits}:{y_limits}")
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1), tile.view(3, -1).std(dim=1)
            s[s == 0] = 1
            tile = transforms.Normalize(mean=m, std=s)(tile)
            tiles.append(tile)

        order_idx = np.random.randint(len(self.permutations))
        permuted_tiels = list(map(lambda x: x[-1], sorted(zip(self.permutations[order_idx], tiles)))) # permuting the tiles
        # data = [tiles[self.permutations[order_idx][t]] for t in range(9)]
        data = torch.stack(permuted_tiels, 0)

        return data, int(order_idx), tiles

    def __len__(self):
        return len(self.names)
