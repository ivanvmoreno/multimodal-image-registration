import os

import numpy as np
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset

class ACROBATDataset(Dataset):
    def __init__(self, source, samples, ext='tiff', transforms=None):
        self.source = source
        self.samples = samples
        self.ext = ext
        self.transforms = transforms
        
    def __getitem__(self, index):
        lx, ly = self.samples[index]
        x = tensor(np.array(Image.open(os.path.join(self.source, f'{lx}.{self.ext}'))))
        y = tensor(np.array(Image.open(os.path.join(self.source, f'{ly}.{self.ext}'))))
        if self.transforms:
            x,y = self.transforms([x,y])
        return x,y

    def __len__(self):
        return len(self.samples)
