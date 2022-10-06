from importlib.resources import path
from pathlib import Path

import webdataset as wds
import torch


from .preprocessing import get_preprocessing_pipeline
from .utils import get_dataset_paths, generate_seqs

import matplotlib.pyplot as plt


def concatenate_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    # Crop the image
    if config['crop']:
        crop_size = 256 - (2 * config['crop_image_resize'][1])
        images = torch.stack(combined_data['jpeg'], dim=0)[:, :, crop_size:, :]

        # Update image resize shape
        config['image_resize'] = [
            1,
            config['crop_image_resize'][1],
            config['crop_image_resize'][2],
        ]

    else:
        images = torch.stack(combined_data['jpeg'], dim=0)

    # Preprocessing
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    # Convert the sequence to input and output
    input_seq = images[0:-1, :, :, :]
    output_seq = images[1:, :, :, :]

    return input_seq, output_seq


def webdataset_data_iterator(config):

    # Get dataset path(s)
    paths = get_dataset_paths(config)

    # Parameter(s)
    BATCH_SIZE = config['BATCH_SIZE']
    SEQ_LEN = config['seq_length'] + config['predict_length']
    number_workers = config['number_workers']

    # Create train, validation, test datasets and save them in a dictionary
    data_iterator = {}

    for key, path in paths.items():
        if path:
            dataset = (
                wds.WebDataset(path, shardshuffle=False)
                .decode("torchrgb")
                .then(generate_seqs, concatenate_samples, SEQ_LEN, config)
            )
            data_loader = wds.WebLoader(
                dataset,
                num_workers=number_workers,
                shuffle=False,
                batch_size=BATCH_SIZE,
            )
            if key in ['training', 'validation']:
                dataset_size = 6250 * len(path)
                data_loader.length = dataset_size // BATCH_SIZE

            data_iterator[key] = data_loader

    return data_iterator
