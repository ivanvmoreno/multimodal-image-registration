from os import listdir
from os.path import isfile, join

import numpy as np
from torch import tensor
from torch.utils.data import Dataset
from PIL import Image


class ACROBATDataset(Dataset):
    def __init__(self, source, samples, ext='tiff', transforms=None):
        self.source = source
        self.samples = samples
        self.ext = ext
        self.transforms = transforms
        
    def __getitem__(self, index):
        lx, ly = self.samples[index]
        x = tensor(np.array(Image.open(join(self.source, f'{lx}.{self.ext}'))))
        y = tensor(np.array(Image.open(join(self.source, f'{ly}.{self.ext}'))))
        if self.transforms:
            x,y = self.transforms([x,y])
        return x,y

    def __len__(self):
        return len(self.samples)


def load_samples_file(path):
    '''Loads a samples file into memory
    
    :param path: Path to the file containing the samples
    :type path: str
    :return: List of samples
    :rtype: list
    '''
    with open(path, 'r') as f:
        return [tuple(_.split(',')) for _ in f.read().splitlines()]


def get_samples_filenames(path, res = None, ext = 'tif'):
    '''Returns a list of sample pairs based on a set of parameters

    :param path: Absolute path to the image samples
    :type path: str
    :param res: Samples resolution. One of {'x5', 'x10'}
    :type res: str
    :param ext: Samples extension. Defaults to 'tif'
    :type ext: str
    :return: List of sample pairs filenames
    :rtype: list
    '''
    flist = [f for f in listdir(path) if isfile(join(path, f)) and f'.{ext}' in f]
    if res is None:
        return flist
    return [f for f in flist if res in f]
    # return [f for f in filenames if res in f and any([f'{t}_' in f for t in types])]


def get_sample_pairs(samples, tag = 'HE', ext = 'auto'):
    '''Returns a list of all possible combinations of sample pairs / groups 
        from a given bag of samples

    :param samples: List of sample samples
    :type samples: list
    :param tag: Tag to be used to identify H&E samples
    :type tag: str
    :param ext: Samples extension. Defaults to 'auto', and assumes homogeneous 
        file extensions for all samples
    :type ext: str
    :return: List of tuples containing sample pairs
    :rtype: list
    '''
    pairs = {}
    comb = []

    # if ext == 'auto':
    #     ext = samples[0].split('.')[-1]

    # Get all sample groups
    for f in samples:
        # Get sample anonimized ID
        uid = f.split('_')[0]
        if uid not in pairs:
            pairs[uid] = [f]
        else:
            pairs[uid].append(f)

    # Generate all sample combinations
    for k, v in pairs.items():
        # Only return paired / grouped samples
        if len(v) >= 2:
            # Store all possible combinations of the group
            comb += [(f'{k}_{tag}', _.split('.')[0]) for _ in v if tag not in _]
            # comb += [(f'{k}_{tag}.{ext}', _) for _ in v if f'{tag}.{ext}' not in _]
    return comb
