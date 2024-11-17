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
IMAGE_PATH = '/home/nsulmol/.local/share/label-studio/media/upload/1'


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


def get_images_filepaths(path: str, ext: str = None) -> list[str]:
    """Given a directory and extension, extract list of image filepaths."""
    files = [os.path.join(path, f) for f in os.listdir(path) if
             os.path.isfile(os.path.join(path, f))]
    if ext:
        files = [f for f in files if ext in f]
    return files


def load_images_from_dir(path: str, ext: str = None) -> list[np.array]:
    """ Loads all images in a directory, normalizes, and returns."""
    file_list = get_images_filepaths(path, str)
    return load_images_from_file_list(file_list)

    # Convert to stacked array
    #return np.stack(tuple(arrs))


def load_images_from_file_list(filepaths: list[str]) -> list[np.array]:
    # Load and convert to numpy
    arrs = []
    for f in filepaths:
        img_src = Image.open(f)
        img_src = np.array(img_src)

        # Whiten before normalizing...
        img_src = scattering.whiten(img_src[np.newaxis, ...])[0, ...]
        img_src = normalize_image(img_src)
        arrs.append(img_src)
    return arrs


def load_labels_images(fname: str, img_path: str
                       ) -> (list[list[int, int, int, int]], list[np.array]):
    """Extract labels and images from JSON labels file (with path to images)."""
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
    filepaths = []
    for img_dict in labelset:
        print(f'Filename: {img_dict[FILENAME_STR]}')
        filepaths.append(os.path.join(img_path, img_dict[FILENAME_STR]))
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

    arrs = load_images_from_file_list(filepaths)
    return total_labels, arrs


def mask_images(arrs: list[np.array], total_labels: list[list[int, int, int, int]]
                ) -> list[np.array]:
    """Mask out labels from all provided arrays.

    Assumes the shape of the first dimension of arrs is the same as the
    length of the first dimension of labels.
    """
    assert(len(arrs) == len(total_labels))

    masked_arrs = []
    for arr, labels in zip(arrs, total_labels):
        arr_shape = arr.shape
        ma_arr = np.ma.masked_array(arr,
                                    mask=np.zeros(arr_shape))
        for label in labels:
            # label: x1x2y1y2
            # Set this label to True (which removes it from masked_array).
            ma_arr.mask[label[2]:label[3], label[0]:label[1]] = True
        masked_arrs.append(ma_arr)
    return masked_arrs

    # Convert to stacked array
    #return np.ma.stack(tuple(masked_arrs))


def save_synthesized(arrs: np.array, path: str, ext: str):
    """Save a numpy array stack in directory path with extension ext."""
    for i in range(0, arrs.shape[0]):
        tmp = convert_to_uint16(arrs[i, ...])
        tmp = Image.fromarray(tmp)
        tmp.save(os.path.join(path, str(i) + ext))


def main():
    #arrs = load_images(INDIR, EXT)
    total_labels, arrs = load_labels_images(os.path.join(INDIR, LABELS_FNAME),
                                            IMAGE_PATH)
    ma_arrs = mask_images(arrs, total_labels)

    # Stack the arrays
    ma_arrs = np.ma.stack(tuple(ma_arrs))
    syns = scattering.synthesis('s_cov_iso', ma_arrs, seed=0,
                                ensemble=True, N_ensemble=N,
                                print_each_step=True)

    save_synthesized(syns, OUTDIR, EXT)


if __name__ == "__main__":
    main()
