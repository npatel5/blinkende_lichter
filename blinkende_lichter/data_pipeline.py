""" Schemas holding data from our pipeline (and the migration code). """
import datajoint as dj
from pipeline import reso, shared, experiment
import numpy as np
from scipy import misc, stats

schema = dj.schema('ecobost_bl', locals())

@schema
class Set(dj.Lookup):
    definition = """  # traininig, validation or test sets
    set       : varchar(8)
    """
    contents = [['train'], ['val'], ['test']]

@schema
class Scan(dj.Computed):
    definition = """ # a single slice (height x width x num_frames) from one scan
    example_id                      : mediumint
    ---
    -> experiment.Scan
    -> shared.Slice
    -> shared.Channel
    -> Set
    """

    @property
    def key_source(self):
        return reso.MaskClassification() # nothing in meso yet

    # INPUT
    class AverageImage(dj.Part):
        definition = """ # p-norm image
        -> master
        ---
        average_image               : longblob
        """

    class CorrelationImage(dj.Part):
        definition = """ # correlation image
        -> master
        ---
        correlation_image           : longblob
        """

    class Sample(dj.Part):
        definition = """ # fifteen minutes sample from the middle of the scan
        -> master
        ---
        filename = ""                : varchar(128)
        """

    class KurtosisImage(dj.Part):
        definition = """ # kurtosis per pixel (unlike average and correlation, calculated on 15-min only)
        -> master
        ---
        kurtosis_image              : longblob
        """

    class SpectralImages(dj.Part):
        definition = """ # average power density for binned frequencies from 0-2.5Hz (calculated on 15-min sample)
        -> master
        ---
        spectral_images              : longblob          # 16 x height x width
        """

    # LABELS
    class WeightedSegmentation(dj.Part):
        definition = """ # Weighted masks per type
        -> master
        -> shared.MaskType
        ---
        weighted_images              : longblob           # masks x height x width
        """

    class Segmentation(dj.Part):
        definition = """ # Segmentations per type. Binarized masks
        -> master
        -> shared.MaskType
        ---
        segmentations                : longblob           # masks x height x width
        """

    class Bboxes(dj.Part):
        definition = """ # x center, y center, height and width of bboxes in the scan. (x, y) = (0, 0) in upper left corner
        -> master
        -> shared.MaskType
        ---
        bboxes                      : longblob            # masks x 4
        """

    def _make_tuples(self, key):
        """ Copy and resize data from pipeline to here and compute what's missing."""
        print('Processing', key)

        # Get some params
        um_height, um_width = (reso.ScanInfo() & key).fetch1('um_height', 'um_width')
        image_height, image_width = (reso.ScanInfo() & key).fetch1('px_height', 'px_width')
        out_height, out_width = int(round(um_height * 2)), int(round(um_width * 2)) # 2 pixels per micron

        # Get next example_id and assign to train/val/test set (80/20/0 for now, we can test in newer scans)
        tuple_ = key.copy()
        tuple_['example_id'] = np.max(self.fetch('example_id')) + 1 if self else 0
        tuple_['set'] = 'train' if np.random.random() < 0.8 else 'val'
        self.insert1(tuple_, ignore_extra_fields=True)

        # Compute average image
        print('Creating average image...')
        avg_image = (reso.SummaryImages.Average() & key).fetch1('average_image')
        upsampled_avg = misc.imresize(avg_image, (out_height, out_width), interp='lanczos', mode='F')
        self.AverageImage().insert1({'example_id':tuple_['example_id'], 'average_image': upsampled_avg})

        # Compute correlation image
        print('Creating correlation image...')
        corr_image = (reso.SummaryImages.Correlation() & key).fetch1('correlation_image')
        upsampled_corr = misc.imresize(corr_image, (out_height, out_width), interp='lanczos', mode='F')
        self.CorrelationImage().insert1({'example_id':tuple_['example_id'], 'correlation_image': upsampled_corr})

        # Get a sample from the middle of the scan
        print('Creating the scan sample...')
        sample = _get_scan_sample(key, sample_length=15, sample_size=(out_height, out_width),
                                  sample_fps=5) # h, w, 4500
        sample_filename = '/mnt/lab/blinkende_lichter/samples/example_{}'.format(tuple_['example_id'])
        np.save(sample_filename, sample)
        self.Sample().insert1({'example_id': tuple_['example_id'], 'filename': sample_filename})

        # Compute kurtosis image
        print('Creating kurtosis image')
        kurtosis_image = stats.kurtosis(sample, axis=-1)
        self.KurtosisImage().insert1({'example_id': tuple_['example_id'], 'kurtosis_image': kurtosis_image})

        # Compute spectral images
        print('Creating spectral images')
        sample -= np.mean(sample, axis=(0, 1)) # subtract overall brightness/drift per frame
        sample -= np.expand_dims(sample.mean(-1), -1) # subtract mean
        freqs = np.abs(np.fft.fft(sample, axis=-1))
        freqs = freqs[:, :, :int(np.ceil(freqs.shape[-1] / 2))]
        spectral_images = [np.mean(chunk, axis=-1) for chunk in np.array_split(freqs, 16, axis=-1)]
        self.SpectralImages().insert1({'example_id': tuple_['example_id'], 'spectral_images': spectral_images})

        # For labels, iterate over different mask types in the scan
        mask_types = np.unique((reso.MaskClassification.Type() & key).fetch('type'))
        for mask_type in mask_types:
            print('Creating segmentation and bboxes for', mask_type)

            # Get weighted masks
            mask_rel = reso.Segmentation.Mask() & key & (reso.MaskClassification.Type() & {'type': mask_type})
            mask_pixels, mask_weights = mask_rel.fetch('pixels', 'weights', order_by='mask_id')
            masks = reso.Segmentation.reshape_masks(mask_pixels, mask_weights, image_height, image_width)
            num_masks = masks.shape[-1]

            weighted_masks = np.empty([num_masks, out_height, out_width])
            for i in range(num_masks):
                weighted_masks[i] = misc.imresize(masks[:, :, i], (out_height, out_width), interp='lanczos', mode='F')
            self.WeightedSegmentation().insert1({'example_id': tuple_['example_id'], 'type': mask_type,
                                                 'weighted_images': weighted_masks})

            # Get segmentations
            segmentations = np.empty([num_masks, out_height, out_width], dtype=np.bool)
            for i in range(num_masks):
                segmentations[i] = _binarize_mask(weighted_masks[i])
            self.Segmentation().insert1({'example_id': tuple_['example_id'], 'type': mask_type,
                                         'segmentations': segmentations})

            # Get bounding boxes
            bboxes = np.empty([num_masks, 4])
            for i in range(num_masks):
                bboxes[i] = _get_bbox(segmentations[i])
            self.Bboxes().insert1({'example_id': tuple_['example_id'], 'type': mask_type, 'bboxes': bboxes})


def _get_scan_sample(key, sample_length=15, sample_size=(-1, -1), sample_fps=5):
    """ Load and correct the scan, get some frames from the middle, resize them and
    interpolate to 5 Hz.

    Arguments:
        key: Dictionary with scan keys including slice and channel.
        length: Length (in minutes) of the sample.
        size: (height, width) Spatial dimensions for the sample.
        fps: Desired frames per second of the sample.
    """
    import scanreader
    from scipy.interpolate import interp1d

    # Read the scan
    scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
    scan = scanreader.read_scan(scan_filename, dtype=np.float32)

    # Load scan
    half_length = round(sample_length/2 * 60 * scan.fps) # 7.5 minutes of recording
    if (scan.num_frames < half_length * 2):
        raise ValueError('Scan {} is too short (< {} min long).'.format(key, sample_length))
    middle_frame = int(np.floor(scan.num_frames / 2))
    frames = slice(middle_frame - half_length, middle_frame + half_length)
    sample = scan[key['slice'] -1, :, :, key['channel'] - 1, frames]
    num_frames = sample.shape[-1]

    # Correct the scan
    correct_raster = (reso.RasterCorrection() & key).get_correct_raster()
    correct_motion = (reso.MotionCorrection() & key).get_correct_motion()
    corrected_sample = correct_motion(correct_raster(sample), frames)

    # Resize
    resized_sample = np.empty([*sample_size, num_frames], dtype=np.float32)
    for i in range(num_frames):
        resized_sample[:, :, i] = misc.imresize(corrected_sample[:, :, i], sample_size, interp='lanczos', mode='F')
    resized_sample=corrected_sample

    # Interpolate to desired frame rate (if memory is a constrain, run per pixel)
    num_output_frames = round(sample_length * 60 * sample_fps)
    f = interp1d(np.linspace(0, 1, num_frames), resized_sample, kind='cubic', copy=False)
    output_sample = f(np.linspace(0, 1, num_output_frames))

    return output_sample


def _binarize_mask(mask, threshold=0.995):
    """ Based on caiman's plot_contours function."""
    mask = mask - mask.min()
    indices = np.unravel_index(np.flip(np.argsort(mask, axis=None), axis=0), mask.shape) # max to min value in mask
    cumulative_mass = np.cumsum(mask[indices]**2) / np.sum(mask**2)
    binary_mask = np.zeros_like(mask, dtype=np.bool)
    binary_mask[tuple(index[cumulative_mass < threshold] for index in indices)] = True

    return binary_mask


def _get_bbox(mask):
    """ Compute x center, y center, height and width of bbox defined by the binary mask."""
    indices = np.where(mask)
    height = np.max(indices[0]) - np.min(indices[0]) + 1
    width = np.max(indices[1]) - np.min(indices[1]) + 1
    y_center = np.min(indices[0]) + height / 2
    x_center = np.min(indices[1]) + width / 2

    return x_center, y_center, height, width