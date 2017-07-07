from glob import glob
from itertools import product
import numpy as np
from scipy.misc import imread
import datajoint as dj
from scipy.signal import convolve2d

from .utils import compute_correlation_image

schema = dj.schema('neurofinder_data', locals())


@schema
class Files(dj.Lookup):
    definition = """
    data_id     : char(5)  # neurofinder id
    type        : enum('train', 'test')
    ---
    path        : varchar(255)
    """

    contents = [
        ('00.00', 'train', '/data/neurofinder.00.00/'),
        ('00.00', 'test', '/data/neurofinder.00.00.test/'),
        ('00.01', 'train', '/data/neurofinder.00.01/'),
        ('00.01', 'test', '/data/neurofinder.00.01.test/'),
        ('00.02', 'train', '/data/neurofinder.00.02/'),
        ('00.03', 'train', '/data/neurofinder.00.03/'),
        ('00.04', 'train', '/data/neurofinder.00.04/'),
        ('00.05', 'train', '/data/neurofinder.00.05/'),
        ('00.06', 'train', '/data/neurofinder.00.06/'),
        ('00.07', 'train', '/data/neurofinder.00.07/'),
        ('00.08', 'train', '/data/neurofinder.00.08/'),
        ('00.09', 'train', '/data/neurofinder.00.09/'),
        ('00.10', 'train', '/data/neurofinder.00.10/'),
        ('00.11', 'train', '/data/neurofinder.00.11/'),
        ('01.00', 'train', '/data/neurofinder.01.00/'),
        ('01.00', 'test', '/data/neurofinder.01.00.test/'),
        ('01.01', 'train', '/data/neurofinder.01.01/'),
        ('01.01', 'test', '/data/neurofinder.01.01.test/'),
        ('02.00', 'train', '/data/neurofinder.02.00/'),
        ('02.00', 'test', '/data/neurofinder.02.00.test/'),
        ('02.01', 'train', '/data/neurofinder.02.01/'),
        ('02.01', 'test', '/data/neurofinder.02.01.test/'),
        ('03.00', 'train', '/data/neurofinder.03.00/'),
        ('03.00', 'test', '/data/neurofinder.03.00.test/'),
        ('04.00', 'train', '/data/neurofinder.04.00/'),
        ('04.00', 'test', '/data/neurofinder.04.00.test/'),
        ('04.01', 'train', '/data/neurofinder.04.01/'),
        ('04.01', 'test', '/data/neurofinder.04.01.test/'),
    ]


@schema
class AveragingParameters(dj.Lookup):
    definition = """
    avg_id              : tinyint   # unique id
    ---
    p                   : float     # power
    contrast_normalize  : bool      # whether to locally contrast normalize or not
    """

    @property
    def contents(self):
        return [(i,) + (p, cn) for i, (p, cn) in enumerate(product([6.], [True, False]))]


@schema
class AverageImage(dj.Imported):
    definition = """
    -> Files
    -> AveragingParameters
    ---
    average_image       : longblob
    """

    def _make_tuples(self, key):
        print('Processing', key, flush=True)
        path = (Files() & key).fetch1('path')
        p, normalize = (AveragingParameters() & key).fetch1('p', 'contrast_normalize')
        files = sorted(glob(path + 'images/*.tiff'))
        scan = np.array([imread(f) for f in files])
        scan = scan.astype(np.float64, copy=False)
        scan = np.power(scan, p, out=scan)  # in place
        average_image = np.sum(scan, axis=0, dtype=np.float64) ** (1 / p)

        if normalize:
            h = np.hamming(71)
            h -= h.min()
            h /= h.sum()
            H = h[:, np.newaxis] * h[np.newaxis, :]
            mu = convolve2d(average_image, H, mode='same', boundary='symm')
            average_image = (average_image - mu) / np.sqrt(
                convolve2d(average_image ** 2, H, mode='same', boundary='symm') - mu ** 2)
        average_image = (average_image - average_image.min()) / (average_image.max() - average_image.min())
        self.insert1(dict(key, average_image=average_image))

@schema
class CorrelationImage(dj.Imported):
    definition = """
    -> Files
    ---
    correlation_image       : longblob
    """

    def _make_tuples(self, key):
        print('Processing', key, flush=True)
        path = (Files() & key).fetch1('path')
        files = sorted(glob(path + 'images/*.tiff'))
        scan = np.array([imread(f) for f in files])
        scan = scan.astype(np.float64, copy=False)
        correlation_image = compute_correlation_image(scan.transpose([2,0,1]))
        self.insert1(dict(key, correlation_image=correlation_image))
