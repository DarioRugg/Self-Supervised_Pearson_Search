from os.path import join
from os import listdir
from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image

img_path = "./../../data/CUHK-SYSU/Image/SSM/s18191.jpg"

# img = Image.open(join(self.data_path, self.names[index])).convert('RGB')
img = read_image(path=img_path)

tiles_limits = lambda dim_size: zip(np.linspace(0, dim_size, 4, dtype=int)[:-1],
                                    np.linspace(0, dim_size, 4, dtype=int)[1:] -
                                    np.linspace(0, dim_size, 4, dtype=int)[:-1])

tiles = []

for (left, width), (top, height) in list(product(tiles_limits(img.shape[1]), tiles_limits(img.shape[2]))):
    tile = transforms.functional.crop(img, top=top, left=left, height=height, width=width)

    # Normalize the patches indipendently to avoid low level features shortcut
    m, s = tile.view(3, -1).mean(dim=1), tile.view(3, -1).std(dim=1)
    s[s == 0] = 1
    tile = transforms.Normalize(mean=m, std=s)(tile)
    tiles.append(tile)

plt.imshow(img.permute(1, 2, 0))

plt.show()


plt.imshow(tiles[0].permute(1, 2, 0))

plt.show()