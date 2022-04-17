import os
import collections


import random
import shutil

from natsort import natsorted

import torch


def nested_dict():
    return collections.defaultdict(nested_dict)


def run_fast_scandir(dir, ext, logs=None):  # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


def get_image_json_files(read_path):
    # Read image files and sort them
    _, file_list = run_fast_scandir(read_path, [".jpeg"])
    image_files = natsorted(file_list)

    # Read json files and sort them
    _, file_list = run_fast_scandir(read_path, [".json"])
    json_files = natsorted(file_list)
    return image_files, json_files


'''
Author:Tai Lei
Date:Thu Nov 22 12:09:27 2018
Info:
'''

# originl transformations
# check: https://github.com/carla-simulator/imitation-learning/issues/1

# from imgaug import augmenters as iaa
# st = lambda aug: iaa.Sometimes(0.4, aug)
# oc = lambda aug: iaa.Sometimes(0.3, aug)
# rl = lambda aug: iaa.Sometimes(0.09, aug)
# seq = iaa.SomeOf((4, None), [
#         # blur images with a sigma between 0 and 1.5
#         rl(iaa.GaussianBlur((0, 1.5))),
#         # add gaussian noise to images
#         rl(iaa.AdditiveGaussianNoise(
#             loc=0,
#             scale=(0.0, 0.05),
#             per_channel=0.5)),
#         # randomly remove up to X% of the pixels
#         oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),
#         # randomly remove up to X% of the pixels
#         oc(iaa.CoarseDropout(
#             (0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)),
#         # change brightness of images (by -X to Y of original value)
#         oc(iaa.Add((-40, 40), per_channel=0.5)),
#         # change brightness of images (X-Y% of original value)
#         st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),
#         # improve or worsen the contrast
#         rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
#         # rl(iaa.Grayscale((0.0, 1))), # put grayscale
# ], random_order=True)


class TransWrapper(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, img):
        return self.seq.augment_image(img)


class RandomTransWrapper(object):
    def __init__(self, seq, p=0.5):
        self.seq = seq
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        return self.seq.augment_image(img)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, id_, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, os.path.join("save_models", "{}_best.pth".format(id_))
        )

