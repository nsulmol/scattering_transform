import os
import json
import numpy as np
from PIL import Image
import fire

# Hack to import scattering
import sys
sys.path.append('./scattering_transform')
import scattering


# NOTE: Kept for convenience
# IMAGE_PATH = '/home/nsulmol/.local/share/label-studio/media/upload/1'


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
    file_list = get_images_filepaths(path, ext)
    return load_images_from_file_list(file_list)


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
    assert len(arrs) == len(total_labels)

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


def extract_objects(arrs: list[np.array], total_labels: list[list[int, int, int, int]]
                ) -> list[np.array]:
    """Exctract labeled objects from all provided arrays and save as list."""
    assert len(arrs) == len(total_labels)

    objects = []
    for arr, labels in zip(arrs, total_labels):
        arr_shape = arr.shape
        for label in labels:
            # label: x1x2y1y2
            objects.append(arr[label[2]:label[3], label[0]:label[1]])
    return objects


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


def synthesize_images_only(img_dir: str, synthesis_count: int,
                           synth_style: str = 's_cov',
                           img_ext: str = '.png') -> np.array:
    """Analyze images and synthesize samples.

    Args:
        img_dir: path to the directory that contains the images to be analyzed.
        synthesis_count: how many images to synthesize.
        img_ext: image file extension, for filtering from img_dir.
        synth_style: str indicating the style of analysis used for synthesis.

    Returns:
        synthesized images.
    """
    arrs = load_images_from_dir(img_dir, img_ext)
    arrs = np.ma.stack(tuple(arrs))
    syns = scattering.synthesis(synth_style, arrs, seed=0,
                                ensemble=True, N_ensemble=synthesis_count,
                                print_each_step=True)
    return syns


def synthesize_images(json_path: str, img_dir: str, synthesis_count: int,
                      synth_style: str = 's_cov') -> np.array:
    """Analyze images with labels removed and synthesize samples.

    Args:
        json_path: path to the label-studio JSON file which contains ROIs for
            each image that has been labeled.
        img_dir: path to the directory that contains the images referenced
            in the JSON file.
        synthesis_count: how many images to synthesize.
        synth_style: str indicating the style of analysis used for synthesis.

    Returns:
        synthesized images.
    """
    total_labels, arrs = load_labels_images(json_path, img_dir)
    ma_arrs = mask_images(arrs, total_labels)

    # Stack the arrays
    ma_arrs = np.ma.stack(tuple(ma_arrs))
    syns = scattering.synthesis(synth_style, ma_arrs, seed=0,
                                ensemble=True, N_ensemble=synthesis_count,
                                print_each_step=True)
    return syns


def synthesize_save_images(json_path: str, img_dir: str, synthesis_count: int,
                           synth_dir: str, synth_style: str = 's_cov',
                           img_ext: str = '.png'):
    """Analyze images with labels removed, synthesize samples and save.

    Args:
        json_path: path to the label-studio JSON file which contains ROIs for
            each image that has been labeled.
        img_dir: path to the directory that contains the images referenced
            in the JSON file.
        synthesis_count: how many images to synthesize.
        synth_dir: path to the directory where the synthesized images will be
            saved.
        synth_style: str indicating the style of analysis used for synthesis.
        img_ext: image file extension, for saving.
    """
    syns = synthesize_images(json_path, img_dir, synthesis_count, synth_style)
    save_arr_stack_as_arrs(syns, synth_dir, img_ext)


def synthesize_save_images_only(img_dir: str, synthesis_count: int,
                                synth_dir: str, synth_style: str = 's_cov',
                                img_ext: str = '.png'):
    """Analyze images, synthesize samples and save.

    Args:
        img_dir: path to the directory that contains the images to be analyzed.
        synthesis_count: how many images to synthesize.
        synth_dir: path to the directory where the synthesized images will be
            saved.
        synth_style: str indicating the style of analysis used for synthesis.
        img_ext: image file extension, for filtering from img_dir and saving.
    """
    syns = synthesize_images_only(img_dir, synthesis_count, synth_style, img_ext)
    save_arr_stack_as_arrs(syns, synth_dir, img_ext)


def save_objects_from_images(json_path: str, img_dir: str, out_dir: str,
                             img_ext: str = '.png'):
    """Extract labeled objects from images and save to dir.

    Args:
        json_path: path to the label-studio JSON file which contains ROIs for
            each image that has been labeled.
        img_dir: path to the directory that contains the images referenced
            in the JSON file.
        out_dir: path to the directory where the object images will be
            saved.
        img_ext: image file extension, for saving.
    """
    total_labels, arrs = load_labels_images(json_path, img_dir)
    objects = extract_objects(arrs, total_labels)
    save_arrs(objects, out_dir, img_ext)


if __name__ == '__main__':
    fire.Fire({
        'synth': synthesize_save_images,
        'synth_images_only': synthesize_save_images_only,
        'objects': save_objects_from_images
    })
