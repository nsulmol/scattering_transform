import os
import json
import numpy as np
import fire

# Hack to import scattering
import sys
sys.path.append('./scattering_transform')
import scattering

from synthesis import utils
import skimage


# NOTE: Kept for convenience
# IMAGE_PATH = '/home/nsulmol/.local/share/label-studio/media/upload/1'


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

    arrs = utils.load_images_from_file_list(filepaths)
    return total_labels, arrs


def mask_images(arrs: list[np.array],
                total_labels: list[list[int, int, int, int]]
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

        # NOTE: torch's MaskedTensor and Complex do not mix well.
        # As a workaround, we are doing inpainting of the image around the
        # masked parts.
        inpainted = skimage.restoration.inpaint_biharmonic(ma_arr.data,
                                                           ma_arr.mask)
        masked_arrs.append(inpainted)
    return masked_arrs


def extract_objects(arrs: list[np.array],
                    total_labels: list[list[int, int, int, int]]
                    ) -> list[np.array]:
    """Exctract labeled objects from all provided arrays and save as list."""
    assert len(arrs) == len(total_labels)

    objects = []
    for arr, labels in zip(arrs, total_labels):
        for label in labels:
            # label: x1x2y1y2
            objects.append(arr[label[2]:label[3], label[0]:label[1]])
    return objects


def _synth_images(arrs: list[np.ndarray], synthesis_count: int,
                  synth_style: str = 's_cov',
                  batch_size: int = None) -> list[np.ndarray]:
    """Synthesis images. Performs n-to-n' from Scattering.

    Args:
        arrs: list of np arrays to evaluate for synthesis.
        synthesis_count: how many images to synthesize.
        synth_style: str indicating the style of analysis used for synthesis.
        batch_size: provide a count to bundle or batch synthesis runs in.
            Useful if you need to synthesize a larger amount than your compute
            memory allows.
    """
    arr_stack = utils.stack_images(arrs)

    syns = []
    i = 0
    if batch_size:
        while synthesis_count >= batch_size:
            print(f'Synthesizing batch {i}')
            synthesis_count -= batch_size
            syns.append(
                scattering.synthesis(synth_style, arr_stack, seed=0,
                                     ensemble=True,
                                     N_ensemble=batch_size,
                                     print_each_step=True))
            i += 1

    if synthesis_count > 0:
        print(f'Synthesizing batch {i}')
        syns.append(
            scattering.synthesis(synth_style, arr_stack, seed=0,
                                 ensemble=True, N_ensemble=synthesis_count,
                                 print_each_step=True))

    # Return to array stack
    syns = np.concatenate(tuple(syns))
    return syns


def synthesize_images_only(img_dir: str, synthesis_count: int,
                           synth_style: str = 's_cov',
                           img_ext: str = '.png',
                           batch_size: int = None) -> np.array:
    """Analyze images and synthesize samples.

    Args:
        img_dir: path to the directory that contains the images to be analyzed.
        synthesis_count: how many images to synthesize.
        img_ext: image file extension, for filtering from img_dir.
        synth_style: str indicating the style of analysis used for synthesis.
        batch_size: provide a count to bundle or batch synthesis runs in.
            Useful if you need to synthesize a larger amount than your compute
            memory allows.

    Returns:
        synthesized images.
    """
    arrs = utils.load_images_from_dir(img_dir, img_ext)
    return _synth_images(arrs, synthesis_count, synth_style, batch_size)


def synthesize_images(json_path: str, img_dir: str, synthesis_count: int,
                      synth_style: str = 's_cov', batch_size: int = None
                      ) -> np.array:
    """Analyze images with labels removed and synthesize samples.

    Args:
        json_path: path to the label-studio JSON file which contains ROIs for
            each image that has been labeled.
        img_dir: path to the directory that contains the images referenced
            in the JSON file.
        synthesis_count: how many images to synthesize.
        synth_style: str indicating the style of analysis used for synthesis.
        batch_size: provide a count to bundle or batch synthesis runs in.
            Useful if you need to synthesize a larger amount than your compute
            memory allows.

    Returns:
        synthesized images.
    """
    total_labels, arrs = load_labels_images(json_path, img_dir)
    ma_arrs = mask_images(arrs, total_labels)
    return _synth_images(ma_arrs, synthesis_count, synth_style, batch_size)


def synthesize_save_images(json_path: str, img_dir: str, synthesis_count: int,
                           synth_dir: str, synth_style: str = 's_cov',
                           img_ext: str = '.png', batch_size: int = None):
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
        batch_size: provide a count to bundle or batch synthesis runs in.
            Useful if you need to synthesize a larger amount than your compute
            memory allows.
    """
    syns = synthesize_images(json_path, img_dir, synthesis_count, synth_style,
                             batch_size)
    utils.save_arr_stack_as_arrs(syns, synth_dir, img_ext)


def synthesize_save_images_only(img_dir: str, synthesis_count: int,
                                synth_dir: str, synth_style: str = 's_cov',
                                img_ext: str = '.png', batch_size: int = None):
    """Analyze images, synthesize samples and save.

    Args:
        img_dir: path to the directory that contains the images to be analyzed.
        synthesis_count: how many images to synthesize.
        synth_dir: path to the directory where the synthesized images will be
            saved.
        synth_style: str indicating the style of analysis used for synthesis.
        img_ext: image file extension, for filtering from img_dir and saving.
        batch_size: provide a count to bundle or batch synthesis runs in.
            Useful if you need to synthesize a larger amount than your compute
            memory allows.
    """
    syns = synthesize_images_only(img_dir, synthesis_count, synth_style,
                                  img_ext, batch_size)
    utils.save_arr_stack_as_arrs(syns, synth_dir, img_ext)


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
    utils.save_arrs(objects, out_dir, img_ext)


if __name__ == '__main__':
    fire.Fire({
        'synth': synthesize_save_images,
        'synth_images_only': synthesize_save_images_only,
        'objects': save_objects_from_images
    })
