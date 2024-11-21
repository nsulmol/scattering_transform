import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
# from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F
import albumentations as A
import random
from skimage.filters import gaussian  # TODO: Consider replacing

from . import utils


class SyntheticSPMDataset(Dataset):
    """ Dataset from merging synthesized 'surfaces' with objects of interest.

    Given a set of synthesized surface images and extracted objects of interest
    (with their masks), we create synthetic images by:
    - Choosing a random number of objects to copy-paste onto the surface of
    size <= max_objects, and:
        * Selecting a randomized set of these objects that matches the number.
        * Expanding the size of all objects so it has dimensions matching the
        expected image dimensions (we place the object in the top-left).
        * Transforming said objects so that they are translated, rotated, and
        scaled randomly.
        * Merging all the masks into a single image. We do this iteratively,
        so that we never overwrite images that have already been placed (i.e.
        intersection cases). Every new image only adds the 'new' pixel data
        (i.e. no intersections).
    - Choosing a random surface and copy-pasting the combined objects image
    onto it (only the masked proportions corresponding to the objects). For
    this, we copy logic from NAME THIS PROJECT.

    Note: for now, we are only considering a single class of objects!

    Attributes:
        surface_filepaths: list of paths to surface files.
        object_filepaths: list of paths to object files.
        max_objects: maximum number of objects to superimpose on a surface
            per item.
        final_transform: additional ablumentation transform to perform on
            image after creating it.
        object_transform: ablumentation transform to perform on objects to
            rotate, scale, and translate them in the image.
    """

    def __init__(self, surfaces_dir: str, objects_dir: str,  # masks_dir: str,
                 max_objects: int, object_scale_limit: tuple[float, float],
                 object_rot_limit: tuple([int, int]),
                 transform: A.Transform = None,
                 img_ext: str = None):
        """Initialize synthetic SPM dataset.

        Args:
            surfaces_dir: directory where the surface images are.
            objects_dir: directory where the object images are.
            max_objects: maximum number of objects to superimpose on a surface
                per item.
            object_scale_limit: min and max scaling to perform on objects when
                augmenting.
            object_rot_limit: min and max rotational range to perform when
                rotating the object (in degrees).
            transform: additional ablumentation transform to perform on final
                image.
            img_ext: image extensions for filtering when reading from
                surfaces_dir or objects_dir.
        """
        self.surface_filepaths = utils.get_images_filepaths(surfaces_dir,
                                                            img_ext)
        self.object_filepaths = utils.get_images_filepaths(objects_dir,
                                                           img_ext)
        # self.masks_dir = masks_dir  # TODO: Remove me?
        self.max_objects = max_objects
        self.final_transform = transform
        self.object_transform = A.ShiftScaleRotate(
            shift_limit=[0.0, 1.0],
            scale_limit=object_scale_limit,
            rotate_limit=object_rot_limit)

    def __len__(self):
        return len(self.surface_filepaths)

    def __getitem__(self, idx):
        surface, objects = self.load_surface_and_objects()
        objects, masks = self.prepare_objects_and_masks(objects, surface.shape)

        # Create compound obj and mask from all objects and masks.
        compound_obj, compound_mask = merge_objects(objects, masks)

        # Create combined image (surface + objects)
        image = image_copy_paste(surface, compound_obj, compound_mask)

        # Run transforms on final image
        if self.transform:
            image = self.transform(image)

        # Wrap sample and targets into torchvision tv_tensors:
        image = tv_tensors.Image(image)
        target = get_target_dict_from_data(objects, masks, idx)
        return image, target

    def load_surface_and_objects(self) -> (np.array, list[np.array]):
        """Load a random surface and objects to superimpose."""
        surface_path = random.choice(self.surface_filepaths)
        num_objects = random.choice(range(0, self.max_objects))
        object_filepaths = [random.choice(self.object_filepaths)
                            for obj in range(num_objects)]

        surface = utils.load_image(surface_path)
        # surface = read_image(surface_path)

        objects = [utils.load_image(path) for path in object_filepaths]
        #objects = [read_image(path) for path in object_filepaths]

        return surface, objects

    def prepare_objects_and_masks(self, objects: list[np.array],
                                  shape: tuple(int, int)
                                  ) -> (list[np.array], list[np.array]):
        """Expand objects to shape and rotate/translate/scale randomly."""
        obj_mask_tuples = [create_widened_object_and_mask(obj, shape)
                           for obj in objects]
        objects = [obj_mask[0] for obj_mask in obj_mask_tuples]
        masks = [obj_mask[1] for obj_mask in obj_mask_tuples]

        # Perform rotation, scaling, and translation on all masks
        transformed_objects = []
        transformed_masks = []
        for obj, mask in zip(objects, masks):
            transformed = self.object_transform(image=obj, mask=mask)
            transformed_objects.append(transformed['image'])
            transformed_masks.append(transformed['mask'])

        return transformed_objects, transformed_masks


def get_target_dict_from_data(objects: list[np.array], masks: list[np.array],
                              image_idx: int) -> dict[str, torch.Tensor]:
    """Create target dictionary from objects and masks.

    Args:
        objects: list of np.array of objects.
        masks: list of np.array of masks.
        image_idx: index to give, int.

    Returns:
        dictionary of (primarily) Tensors, containing:
        - 'boxes': torchvision.tv_tensors.BoundingBoxes object containing all
        the object bounding boxes.
        - 'labels: Tensor of shape [N] containing the class label of each box.
        - 'image_id': int containing a unique ID associated to this image.
        - 'area': Tensor of shape [N] containing the area of each bounding box.
        - 'iscrowd': Tensor of shape [N], bool for each box. Those  with True
        will be ignored during evaluation.
        - 'masks': torchvision.tv_tensors.Mask of shape [N, H, W] with
        segmentation masks for each object.
    """

    # Extract bounding boxes ???
    # boxes = [bounding_box_from_mask(mask) for mask in masks]
    boxes = [masks_to_boxes(mask) for mask in masks]
    boxes = torch.cat(boxes)  # Concatenate all masks
    num_objs = boxes.shape(0)

    # there is only one class
    labels = torch.ones((num_objs,), dtype=torch.int64)

    # Get areas for each box
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # suppose all instances are not crowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    # Package up data in dict and send out
    target = {}
    target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY",
                                               canvas_size=F.get_size(img))
    target["masks"] = tv_tensors.Mask(masks)
    target["labels"] = labels
    target["image_id"] = image_idx
    target["area"] = area
    target["iscrowd"] = iscrowd

    return target


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    """TAKEN FROM conradry/copy-paste-aug on github.

    Here, alpha is presumably the combined mask?
    """
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]
        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)

    return img


def create_widened_object_and_mask(obj: np.array,
                                   shape: tuple(int, int)
                                   ) -> tuple(np.array, np.array):
    """Expand arr right and down with zeros, and create mask."""
    bigger_obj = np.zeros(shape, obj.dtype)
    bigger_obj[0:obj.shape[0], 0:obj.shape[1]] = obj
    mask = np.zeros(shape, obj.dtype)
    mask[0:obj.shape[0], 0:obj.shape[1]] = 1  # True
    return bigger_obj, mask


def merge_objects(objects: list[np.array], masks: list[np.array]
                  ) -> tuple(np.array, np.array):
    """Merge all objects and masks into a single object and mask."""
    final_obj = objects[0]
    final_mask = masks[0]

    for obj, mask in zip(objects[1:], masks[1:]):
        # Logical NOR between final mask and current.
        # (This is a flipped AND, because in np.ma 1 means don't consider).
        final_mask = ~np.logical_or(final_mask, mask)
        # Add only the masked portions between combined obj and
        # current obj.
        final_obj = np.ma.masked_array(final_obj.data, obj.data,
                                        final_mask).data
    return final_obj, final_mask


def bounding_box_from_mask(mask: np.array) -> list[int, int, int, int]:
    """Extract bounding box coordinates from a mask containing one object.

    Args:
        mask: np.array mask containing one object.

    Returns:
        bounding box as list of ints [x1, y1, x2, y2].
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax, rmax
