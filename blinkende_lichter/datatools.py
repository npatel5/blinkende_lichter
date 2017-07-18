import random
import numpy as np
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset

class ListDataset(Dataset):
    """
    Arguments:
        *data (indexable): datasets
    """

    def __init__(self, *data, transform=None):
        self.transform = transform
        for d in data:
            assert len(d) == len(data[0]), 'datasets must have same first dimension'
        self.data = data

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(tuple(d[index] for d in self.data))
        else:
            return tuple(d[index] for d in self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return '\n'.join(['List  {}: {}'.format(i, str(len(t))) for i, t in enumerate(self.data)])

class RandomFlip:
    def __call__(self, img):
        if random.random() > 0.5:
            row_slice = slice(None, None, -1)
        else:
            row_slice = slice(0, None)

        if random.random() > 0.5:
            col_slice = slice(None, None, -1)
        else:
            col_slice = slice(0, None)

        if isinstance(img, np.ndarray):
            return img[..., row_slice, col_slice]
        else:
            return tuple(im[..., row_slice, col_slice] for im in img)

class TypeConversion:

    def __init__(self, *dtypes):
        self.dtypes = dtypes

    def __call__(self, x):
        return tuple(e.astype(dtype) for e, dtype in zip(x, self.dtypes))

