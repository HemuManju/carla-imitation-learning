import numpy as np
from skimage.io import imread_collection

from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .utils import nested_dict


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
    def __init__(self, data):
        super(TorchDataset, self).__init__()
        self.x = data['x']
        self.y = data['y']

        # This step normalizes image between 0 and 1
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        # Read only specific data and convert to torch tensors
        x = self.transform(self.x[index]).type(torch.float32)
        y = torch.from_numpy(self.y[index]).type(torch.long)
        return x, y.squeeze(-1)

    def __len__(self):
        return self.x.shape[0]


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
    data = get_data[data_split_type](hparams)

    # Create train, validation, test datasets and save them in a dictionary
    data_iterator = {}
    train_data = TorchDataset(data['train'])
    data_iterator['train_dataloader'] = DataLoader(train_data,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)

    valid_data = TorchDataset(data['valid'])
    data_iterator['val_dataloader'] = DataLoader(valid_data,
                                                 batch_size=BATCH_SIZE)

    test_data = TorchDataset(data['test'])
    data_iterator['test_dataloader'] = DataLoader(test_data,
                                                  batch_size=BATCH_SIZE)

    return data_iterator


def continous_to_discreet(y):
    steer = y[:, 1].copy()
    # Discretize
    steer[y[:, 1] > 0.05] = 2.0
    steer[y[:, 1] < -0.05] = 0.0
    steer[~np.logical_or(steer == 0.0, steer == 2.0)] = 1.0

    # Discretize throttle and brake
    throttle = y[:, 0]
    brake = y[:, 2]

    acc = brake.copy()
    acc[np.logical_and(brake == 0.0, throttle == 1.0)] = 2.0
    acc[np.logical_and(brake == 0.0, throttle == 0.5)] = 1.0
    acc[np.logical_and(brake == 1.0, throttle == 0.0)] = 0.0

    actions = np.vstack((acc, steer)).T
    # Convert actions to indices
    action_ind = actions[:, 0] * 3 + actions[:, 1]

    return action_ind


def get_pooled_data(hparams):

    read_paths = []
    action_ind = []
    camera = hparams['camera']
    for log in hparams['train_logs']:
        read_paths.append(hparams['data_dir'] + 'raw' + '/' + log + '/' +
                          camera + '/*.jpeg')
        autopilot_actions = np.genfromtxt(hparams['data_dir'] + 'raw' + '/' +
                                          log + '/state.csv',
                                          delimiter=',',
                                          usecols=(4, 5, 6, 7))
        action_ind.append(continous_to_discreet(autopilot_actions))

    actions = np.stack(action_ind, axis=-1)
    images = imread_collection(read_paths).concatenate()

    # Convert to gray scale
    images = np.dot(images[..., :], [0.299, 0.587, 0.114])

    # Split train, validation, and testing
    test_size = hparams['TEST_SIZE']
    x = np.arange(actions.shape[0])
    train_id, val_id, test_id = np.split(
        x,
        [int((1 - 2 * (test_size)) * len(x)),
         int((1 - test_size) * len(x))])

    # Data
    data = nested_dict()
    data['train']['x'] = images[train_id]
    data['train']['y'] = actions[train_id]

    data['valid']['x'] = images[val_id]
    data['valid']['y'] = actions[val_id]

    data['test']['x'] = images[test_id]
    data['test']['y'] = actions[test_id]

    return data


def get_leave_out_data(hparams):
    read_paths = []
    camera = hparams['camera']
    for log in hparams['train_logs']:
        read_paths.append(hparams['data_dir'] + 'raw' + '/' + log + '/' +
                          camera + '/*.jpeg')
    images = imread_collection(read_paths).concatenate()

    # Autopilot actions

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
                          camera + '/*.jpeg')
    test_data = imread_collection(read_paths).concatenate()

    # Data
    data = nested_dict()

    # Data
    data = nested_dict()
    data['train']['x'] = train_data
    data['train']['y'] = None

    data['val']['x'] = val_data
    data['val']['y'] = None

    data['test']['x'] = test_data
    data['test']['y'] = None

    return train_data, val_data, test_data
