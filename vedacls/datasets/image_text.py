import os

import cv2

from .base import BaseDataset
from .registry import DATASETS


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(txt, extensions=None, is_valid_file=None):
    images = []
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    lines = open(txt, 'r').readlines()
    for line in lines:
        path, label = line.split(' ')
        if is_valid_file(path):
            item = (path, int(label))
            images.append(item)

    return images


@DATASETS.register_module
class ImageText(BaseDataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (image path, class_index) tuples
    """

    def __init__(self, text_path, prefix='', transform=None, target_transform=None):
        super(ImageText, self).__init__(transform, target_transform)

        self.samples = make_dataset(text_path, IMG_EXTENSIONS)
        self.prefix = prefix

    def __getitem__(self, idx):
        image_file, label = self.samples[idx]

        image = cv2.imread(os.path.join(self.prefix, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_process(image)
        label = self.target_process(label)

        return image, label, image_file

    def __len__(self):
        return len(self.samples)
