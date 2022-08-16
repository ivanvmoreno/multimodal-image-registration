from os import listdir
from os.path import isfile, join
from random import shuffle

import numpy as np
import torch
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(self, filenames, mode='train'):
        self.mode = mode
        self.pair = get_sample_pairs(filenames)
        shuffle(self.pair)

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, index):
        self.data_A = np.load(self.pair[index][0])
        self.data_B = np.load(self.pair[index][1])
        return torch.Tensor(self.data_A), torch.Tensor(self.data_B)


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


def get_sample_pairs(filenames):
    '''Returns a list of sample pairs from a given bag of samples

    :param filenames: List of sample filenames
    :type filenames: list
    :return: List of tuples containing sample pairs filenames
    :rtype: list
    '''
    sample_ids = [f.split('_')[0] for f in filenames]
    pairs = [tuple([f for f in filenames if f'{id}_' in f]) for id in sample_ids]
    return [p for p in pairs if len(p) == 2]
