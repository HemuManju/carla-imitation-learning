from pathlib import Path
import natsort

import webdataset as wds
import torch

from torchvision import transforms

from itertools import islice, cycle

preproc = transforms.Compose(
    [transforms.Grayscale(), transforms.Normalize(mean=[0.5], std=[0.5])]
)


def concatenate_samples(samples):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = transforms.functional.rotate(
        torch.stack(combined_data['jpeg'], dim=0), angle=90
    )
    images = preproc(images).squeeze(1)
    last_data = samples[-1]['json']

    if last_data['direction'] in [-1, 5, 6]:
        command = 4
    else:
        command = last_data['direction']

    action = torch.tensor(
        [last_data['throttle'], last_data['steer'], last_data['brake']]
    )
    return images, command, action


def generate_seqs(src, nsamples=3):
    it = iter(src)
    result = tuple(islice(it, nsamples))
    if len(result) == nsamples:
        yield concatenate_samples(result)
    for elem in it:
        result = result[1:] + (elem,)
        yield concatenate_samples(result)


def find_tar_files(read_path, pattern):
    files = [str(f) for f in Path(read_path).glob('*.tar') if f.match(pattern + '*')]
    return natsort.natsorted(files)


def get_dataset_paths(config):
    paths = {}
    data_split = config['data_split']
    read_path = config['raw_data_path']
    for key, split in data_split.items():
        combinations = [
            '_'.join(item)
            for item in zip(
                cycle(split['town']), split['season'], cycle(split['behavior'])
            )
        ]

        # Get all the tar files
        temp = [find_tar_files(read_path, combination) for combination in combinations]

        # Concatenate all the paths and assign to dict
        paths[key] = sum(temp, [])  # Not a good way, but it is fun!
    return paths


def webdataset_data_iterator(config):

    paths = get_dataset_paths(config)

    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    SEQ_LEN = config['obs_size']
    number_workers = config['number_workers']

    # Create train, validation, test datasets and save them in a dictionary
    data_iterator = {}

    for key, path in paths.items():
        if path:
            dataset = (
                wds.WebDataset(path, shardshuffle=False)
                .decode("torchrgb")
                .pipe(generate_seqs, SEQ_LEN)
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
