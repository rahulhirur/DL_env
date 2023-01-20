from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd



class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    
    def __init__(self, data, mode: str):
        self.data = data
        self.mode= mode
        train_mean = [0.59685254, 0.59685254, 0.59685254]
        train_std = [0.16043035, 0.16043035, 0.16043035]
        if mode == 'train':
            self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean =train_mean, std= train_std, inplace=False)])
        elif mode == 'val':
            self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean =train_mean, std= train_std, inplace=False)])
        else:
            print('Invalid mode given')
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self._transform(gray2rgb(imread(self.data.filename[index])))
        labels =torch.tensor((self.data.crack[index], self.data.inactive[index]))
        return (img, labels)
        
    
    pass
