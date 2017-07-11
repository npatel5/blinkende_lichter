import json
from glob import glob
from itertools import product
import numpy as np
from scipy.misc import imread, imresize
import datajoint as dj
from scipy.signal import convolve2d

from .utils import compute_correlation_image, to_mask

schema = dj.schema('neurofinder_data', locals())


class Upsample:
    def fetch_upsampled(self, px_per_mu):
        img_attr = [k for k, v in self.heading.attributes if v.type == 'longblob']
        assert len(img_attr) == 1, 'Cannot determine name of image attribute'
        img_attr = img_attr[0]
        keys, tmp, resolutions = (self * ScanInfo()).fetch(dj.key, img_attr, 'resolution')
        ret = []
        for t, r in zip(tmp, resolutions):
            if len(t.shape) < 3:
                t = imresize(t, size=px_per_mu/r)
            else:
                t = np.array([imresize(tt, size=px_per_mu/r) for tt in t])
            ret.append(t)
        return keys, ret


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
class ScanInfo(dj.Imported):
    definition = """
    -> Files
    ---
    width       : smallint  # image with
    height      : smallint  # image height
    nframes     : int       # number of frames
    indicator   : varchar(50)
    frame_rate  : float     # in Hz
    region      : varchar(50)
    lab         : varchar(50)   
    resolution  : float     # resolution in pixels per micron
    """

    def _make_tuples(self, key):
        print('Processing', key, flush=True)
        path = (Files() & key).fetch1('path')
        with open(path + 'info.json') as fid:
            info = json.load(fid)
        self.insert1(dict(key,
                          width=info['dimensions'][0],
                          height=info['dimensions'][1],
                          nframes=info['dimensions'][2],
                          indicator=info['indicator'],
                          frame_rate=info['rate-hz'],
                          region=info['region'],
                          lab=info['lab'],
                          resolution=info['pixels-per-micron'],
                          ))


@schema
class Segmentation(dj.Imported, Upsample):
    definition = """
    -> ScanInfo
    ---
    masks       : longblob  # neurons x width x height array
    """

    @property
    def key_source(self):
        return Files() & dict(type='train')

    def _make_tuples(self, key):
        print('Populating', key, flush=True)
        # load the regions (training data only)
        path = (Files() & key).fetch1('path')
        with open(path + 'regions/regions.json') as f:
            regions = json.load(f)

        dims = (ScanInfo() & key).fetch1('width', 'height')
        key['masks'] = np.array([to_mask(s['coordinates'], dims) for s in regions])
        self.insert1(key)


@schema
class AverageImage(dj.Imported, Upsample):
    definition = """
    -> ScanInfo
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
class CorrelationImage(dj.Imported, Upsample):
    definition = """
    -> ScanInfo
    ---
    correlation_image       : longblob
    """

    def _make_tuples(self, key):
        print('Processing', key, flush=True)
        path = (Files() & key).fetch1('path')
        files = sorted(glob(path + 'images/*.tiff'))
        scan = np.array([imread(f) for f in files])
        scan = scan.astype(np.float64, copy=False)
        correlation_image = compute_correlation_image(scan.transpose([1, 2, 0]))
        self.insert1(dict(key, correlation_image=correlation_image))


@schema
class SpectralImage(dj.Imported, Upsample):
    definition = """
    -> ScanInfo
    ---
    spectral_image       : longblob
    """

    def _make_tuples(self, key):
        print('Processing', key, flush=True)
        path = (Files() & key).fetch1('path')
        fr, T = (ScanInfo() & key).fetch1('frame_rate', 'nframes')

        files = sorted(glob(path + 'images/*.tiff'))
        scan = np.array([imread(f) for f in files], dtype=np.float32)
        F = np.abs(np.fft.fft(scan, axis=0))

        w = np.fft.fftfreq(T, d=1. / fr)
        ti = np.interp(np.linspace(1e-6, 1., 16), w[1:T // 2], np.arange(1, T // 2)).astype(int)

        spectral_image = []
        for t1, t2 in zip(ti[:-1], ti[1:]):
            spectral_image.append(F[t1:t2, ...].sum(axis=0))
        spectral_image = np.array(spectral_image)
        spectral_image = spectral_image / spectral_image.sum(axis=0, keepdims=True)

        self.insert1(dict(key, spectral_image=spectral_image))
