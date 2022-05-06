import os
import collections

from natsort import natsorted

from torchvision import transforms


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


def get_preprocessing_pipeline(config):
    preproc = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(
                size=(config['image_resize'][1], config['image_resize'][2])
            ),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return preproc
