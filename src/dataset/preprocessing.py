from torchvision import transforms


import numpy as np
import cv2
from numpy.polynomial import Polynomial as P

import matplotlib.pyplot as plt


class CropTransform:
    """Rotate by one of the given angles."""

    def __init__(self, top=0, left=0, height=0, width=0):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, x):
        return transforms.functional.crop(
            x, self.top, self.left, self.height, self.width
        )


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angles)


def get_preprocessing_pipeline(config):

    preproc = transforms.Compose(
        [
            RotationTransform(angles=-90),
            transforms.Grayscale(),
            transforms.Resize(
                size=(config['image_resize'][1], config['image_resize'][2])
            ),
            # transforms.Normalize(mean=[0.5], std=[1.0]),
            # transforms.ToTensor(),
        ]
    )
    return preproc
