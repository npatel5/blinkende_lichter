import random
from torch.utils.data import Dataset

class RandomFlip:
    def __call__(self, img):
        if random.random() > 0.5:
            row_slice = slice(None, None, -1)
        else:
            row_slice = slice(None, None, None)

        if random.random() > 0.5:
            col_slice = slice(None, None, -1)
        else:
            col_slice = slice(None, None, None)
        return img[..., row_slice, col_slice]


