import numpy as np
from skimage.io import imread_collection

from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TorchDataset(Dataset):
    """All subject dataset class.

    Parameters
    ----------
    split_ids : list
        ids list of training or validation or traning data.

    Attributes
    ----------
    split_ids

    """
    def __init__(self, images):
        super(TorchDataset, self).__init__()
        self.images = images

        # This step normalizes image between 0 and 1
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        # Read only specific data and convert to torch tensors
        x = self.transform(self.images[index]).type(torch.float32)
        return x

    def __len__(self):
        return self.images.shape[0]


def train_val_test_iterator(hparams, data_split_type=None):
    """A function to get train, validation, and test data.

    Parameters
    ----------
    hparams : yaml
        The hparamsuration file.
    leave_out : bool
        Whether to leave out some subjects training and use them in testing

    Returns
    -------
    dict
        A dict containing the train and test data.

    """
    # Parameters
    BATCH_SIZE = hparams['BATCH_SIZE']

    # Get training, validation, and testing data
    get_data = {
        'pooled_data': get_pooled_data,
        'leave_one_out_data': get_leave_out_data
    }
    train_data, valid_data, test_data = get_data[data_split_type](hparams)

    # Create train, validation, test datasets and save them in a dictionary
    data_iterator = {}
    train_data = TorchDataset(train_data)
    data_iterator['train_dataloader'] = DataLoader(train_data,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)

    valid_data = TorchDataset(valid_data)
    data_iterator['val_dataloader'] = DataLoader(valid_data,
                                                 batch_size=BATCH_SIZE)

    test_data = TorchDataset(test_data)
    data_iterator['test_dataloader'] = DataLoader(test_data,
                                                  batch_size=BATCH_SIZE)

    return data_iterator


def get_pooled_data(hparams):
    read_paths = []
    camera = hparams['camera']
    for log in hparams['train_logs']:
        read_paths.append(hparams['data_dir'] + 'raw' + '/' + log + '/' +
                          camera + '_resized_224_bw' + '/*.png')
    images = imread_collection(read_paths).concatenate()

    # Split train and validation
    test_size = hparams['TEST_SIZE']
    ids = np.arange(len(images))
    train_id, test_id, _, _ = train_test_split(ids,
                                               ids * 0,
                                               test_size=test_size,
                                               shuffle=True)

    # Data containers
    train_data = images[train_id]
    test_data = images[test_id]

    # Split val from train data
    val_size = hparams['VAL_SIZE']
    ids = np.arange(len(train_data))
    train_id, val_id, _, _ = train_test_split(ids,
                                              ids * 0,
                                              test_size=val_size,
                                              shuffle=True)

    val_data = train_data[val_id]
    train_data = train_data[train_id]

    return train_data, val_data, test_data


def get_leave_out_data(hparams):
    read_paths = []
    camera = hparams['camera']
    for log in hparams['train_logs']:
        read_paths.append(hparams['data_dir'] + 'raw' + '/' + log + '/' +
                          camera + '_resized_224_bw' + '/*.png')
    images = imread_collection(read_paths).concatenate()

    # Split train and validation
    val_size = hparams['VALID_SIZE']
    ids = np.arange(len(images))
    train_id, val_id, _, _ = train_test_split(ids,
                                              ids * 0,
                                              test_size=val_size,
                                              shuffle=True)
    train_data = images[train_id]
    val_data = images[val_id]

    # Testing data
    read_paths = []
    for log in hparams['test_logs']:
        read_paths.append(hparams['data_dir'] + 'raw' + '/' + log + '/' +
                          camera + '_resized_224_bw' + '/*.png')
    test_data = imread_collection(read_paths).concatenate()

    return train_data, val_data, test_data
