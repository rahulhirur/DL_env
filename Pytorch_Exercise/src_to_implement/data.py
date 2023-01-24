from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd

class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + t.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description

    def __init__(self, data, mode: str):
        self.data = data
        self.mode = mode
        train_mean = [0.59685254, 0.59685254, 0.59685254]
        train_std = [0.16043035, 0.16043035, 0.16043035]
        if mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomApply([tv.transforms.ColorJitter(brightness=0, contrast=0.1, saturation=0, hue=0)], p=0.3),
                tv.transforms.ToTensor(),

                tv.transforms.RandomHorizontalFlip(p=0.3),
                tv.transforms.RandomVerticalFlip(p=0.3),
                tv.transforms.RandomErasing(p=0.3, ratio=(0.54, 0.4), scale=(0.02, 0.04), value=0),
                tv.transforms.RandomApply([tv.transforms.Lambda(AddGaussianNoise(0, .009))], p=0.1),
                tv.transforms.Normalize(mean=train_mean, std=train_std, inplace=False)])

        elif mode == 'val':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std, inplace=False)])
        else:
            print('Invalid mode given')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self._transform(gray2rgb(imread(self.data.iloc[index, 0])))
        labels = torch.tensor((self.data.iloc[index, 1], self.data.iloc[index, 2]))
        return img, labels
