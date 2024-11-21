import os
import numpy as np
from PIL import Image

# Hack to import scattering
import sys
sys.path.append('./scattering_transform')
import scattering


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


def load_image(path: str) -> np.array:
    """Load image, whiten, and normalize."""
    img_src = Image.open(path)
    img_src = np.array(img_src)

    # Whiten before normalizing...
    img_src = scattering.whiten(img_src[np.newaxis, ...])[0, ...]
    img_src = normalize_image(img_src)

    return img_src


def load_images_from_dir(path: str, ext: str = None) -> list[np.array]:
    """Load all images in a directory, normalizes, and returns."""
    file_list = get_images_filepaths(path, ext)
    return load_images_from_file_list(file_list)


def load_images_from_file_list(filepaths: list[str]) -> list[np.array]:
    """Load images from a provided list of filepaths."""
    # arrs = []
    # for f in filepaths:
    #     img_src = load_image(f)
    #     arrs.append(img_src)
    # return arrs
    return [load_image(f) for f in filepaths]


def save_arrs(arrs: list[np.array], path: str, ext: str):
    """Save a list of numpy arrays to directory path with extension ext."""
    for i in range(0, len(arrs)):
        tmp = convert_to_uint16(arrs[i])
        tmp = Image.fromarray(tmp)
        tmp.save(os.path.join(path, str(i) + ext))


def save_arr_stack_as_arrs(arrs: np.array, path: str, ext: str):
    """Save a numpy array stack in directory path with extension ext."""
    for i in range(0, arrs.shape[0]):
        tmp = convert_to_uint16(arrs[i, ...])
        tmp = Image.fromarray(tmp)
        tmp.save(os.path.join(path, str(i) + ext))
