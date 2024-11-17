import os
import json
import numpy as np
from PIL import Image

# Hack to import scattering
import sys
sys.path.append('./scattering_transform')
import scattering

import matplotlib.pyplot as plt

# Parameters
N = 10  # How many synthesis images to create
INDIR = './data/'  # './data/'
OUTDIR = './syn/'  # './syn/'
EXT = '.png'
LABELS_FNAME = 'labels.json'


def normalize_image(arr: np.array) -> np.array:
    """Min-max normalize an image."""
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr


def convert_to_float(arr: np.array) -> np.array:
    """Convert arr to float."""
    return arr.astype(float)  # Convert to float


def convert_to_uint16(arr: np.array) -> np.array:
    """Convert array back to uint, scaling accordingly."""
    arr = arr * 65535  # HARD-CODED VALUE!!!
    return arr.astype(np.uint16)


def load_images(path: str, ext: str = None) -> np.array:
    """ Loads all images in a directory, normalizes, and stacks."""
    # Get files
    files = [f for f in os.listdir(path) if
             os.path.isfile(os.path.join(path, f))]

    if ext:
        files = [f for f in files if ext in f]

    # Load and convert to numpy
    arrs = []
    for f in files:
        img_src = Image.open(os.path.join(path, f))
        img_src = np.array(img_src)

        # Whiten before normalizing...
        img_src = scattering.whiten(img_src[np.newaxis, ...])[0, ...]

        # print(f'img_src dtype: {img_src.dtype}, shape: {img_src.shape}')
        img_src = normalize_image(img_src)

        # print(f'img_src dtype: {img_src.dtype}, shape: {img_src.shape}')

        # print(f'img_src dtype: {img_src.dtype}, shape: {img_src.shape}')
        # plt.imshow(img_src, interpolation='nearest')
        # plt.show(block=True)

        arrs.append(img_src)
        # print(f'arrs len: {len(arrs)}')

    # print(f'arrs len: {len(arrs)}')

    # Convert to stacked array
    return np.stack(tuple(arrs))


def load_labels(fname: str) -> list[list[int, int, int, int]]:
    """Extract labels from JSON labels file."""
    # Hardcoded crud (for now)
    ANNOTATIONS_STR = 'annotations'
    RESULT_STR = 'result'
    ORIG_WIDTH_STR = 'original_width'
    ORIG_HEIGHT_STR = 'original_height'
    VALUE_STR = 'value'
    X_STR = 'x'
    Y_STR = 'y'
    WIDTH_STR = 'width'
    HEIGHT_STR = 'height'
    FILENAME_STR = 'file_upload'

    print('Loading labels...')
    with open(fname) as f:
        labelset = json.load(f)

    total_labels = []  # Will hold x1, x2, y1, y2
    for img_dict in labelset:
        print(f'Filename: {img_dict[FILENAME_STR]}')
        img_labels = []

        annotations = img_dict[ANNOTATIONS_STR][0][RESULT_STR]
        # print(f'Annotations: {annotations}')
        for annotation in annotations:
            w = annotation[ORIG_WIDTH_STR]
            h = annotation[ORIG_HEIGHT_STR]
            values = annotation[VALUE_STR]

            x1x2y1y2 = [int(w * values[X_STR] / 100),
                        int(w * values[X_STR] / 100 +
                            w * values[WIDTH_STR] / 100),
                        int(h * values[Y_STR] / 100),
                        int(h * values[Y_STR] / 100 +
                            h * values[HEIGHT_STR] / 100)]
            img_labels.append(x1x2y1y2)
        total_labels.append(img_labels)
    return total_labels


def mask_images(arrs: np.array, total_labels: list[list[int, int, int, int]]
                ) -> np.array:
    """Mask out labels from all provided arrays.

    Assumes the shape of the first dimension of arrs is the same as the
    length of the first dimension of labels.
    """
    assert(arrs.shape[0] == len(total_labels))

    masked_arrs = []
    for i in range(0, len(total_labels)):
        arr = arrs[i, ...]
        arr_shape = arr.shape
        ma_arr = np.ma.masked_array(arr,
                                    mask=np.zeros(arr_shape))

        for label in total_labels[i]:
            # label: x1x2y1y2
            # Set this label to True (which removes it from masked_array).

            # print(f'label: {label}')
            ma_arr.mask[label[2]:label[3], label[0]:label[1]] = True

        masked_arrs.append(ma_arr)

        # print('showing masked array...')
        # plt.imshow(ma_arr, interpolation='nearest')
        # plt.show(block=True)

    # Convert to stacked array
    # return np.stack(tuple(masked_arrs))
    return np.ma.stack(tuple(masked_arrs))


def save_synthesized(arrs: np.array, path: str, ext: str):
    """Save a numpy array stack in directory path with extension ext."""
    for i in range(0, arrs.shape[0]):
        tmp = convert_to_uint16(arrs[i, ...])
        tmp = Image.fromarray(tmp)
        tmp.save(os.path.join(path, str(i) + ext))


def main():
    arrs = load_images(INDIR, EXT)
    total_labels = load_labels(os.path.join(INDIR, LABELS_FNAME))
    ma_arrs = mask_images(arrs, total_labels)

    # print(ma_arrs[0])
    # print(ma_arrs[0].mask)

    # print('showing masked array...')
    # plt.imshow(ma_arrs[0], interpolation='nearest')
    # plt.show(block=True)

    # plt.imshow(ma_arrs[0].mask, interpolation='nearest')
    # plt.show(block=True)

    # print('showing NOT masked array...')
    # plt.imshow(arrs[0], interpolation='nearest')
    # plt.show(block=True)

    # print(f'arrs: shape: {arrs.shape}')
    # print(f'total_labels: len: {len(total_labels)}')
    # print(f'ma_arrs: shape: {ma_arrs.shape}')

    # Whiten data  --- Does not work!
    # arrs = scattering.whiten(arrs, overall=True)

    syns = scattering.synthesis('s_cov', ma_arrs, seed=0,
                                ensemble=True, N_ensemble=N,
                                print_each_step=True)

    save_synthesized(syns, OUTDIR, EXT)


if __name__ == "__main__":
    main()
