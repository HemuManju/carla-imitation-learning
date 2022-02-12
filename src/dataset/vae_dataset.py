from PIL import Image

import torch
from torchvision import transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .utils import get_image_json_files


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


class VAETorchDataset(Dataset):
    def __init__(self, read_path, config=None):
        """
        Args:
            path (str): path to the dataset folder
            logs (list): list of log folders
            cameras (list): list of camera folders
            transform (torchvision.transforms): transforms to apply to image
        """
        # Compose transforms
        self.transform = transforms.ToTensor()

        self.image_files, json_files = get_image_json_files(read_path=read_path)

    def __getitem__(self, index):
        """Get single image."""
        return self.transform(Image.open(self.image_files[index]).convert('L'))

    def __len__(self):
        """Return dataset length."""
        return len(self.image_files)


class SeqVAETorchDataset(Dataset):
    def __init__(self, read_path, config=None):
        """
        Args:
            path (str): path to the dataset folder
            logs (list): list of log folders
            cameras (list): list of camera folders
            transform (torchvision.transforms): transforms to apply to image
        """
        # Compose transforms
        self.transform = transforms.ToTensor()

        self.image_files, json_files = get_image_json_files(read_path=read_path)

    def __getitem__(self, index):
        """Get single image."""
        return self.transform(Image.open(self.image_files[index]).convert('L'))

    def __len__(self):
        """Return dataset length."""
        return len(self.image_files)


def train_val_test_iterator(config, dataset_type='individual'):
    """A function to get train, validation, and test data.

    Parameters
    ----------
    config : yaml
        The hparamsuration file.
    leave_out : bool
        Whether to leave out some subjects training and use them in testing

    Returns
    -------
    dict
        A dict containing the train and test data.

    """
    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    dataset = {'individual': VAETorchDataset, 'sequential': SeqVAETorchDataset}

    # Create train, validation, test datasets and save them in a dictionary
    data_iterator = {}
    train_data = dataset[dataset_type](read_path=config['train_data_path'])
    data_iterator['train_data_loader'] = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    valid_data = dataset[dataset_type](read_path=config['val_data_path'])
    data_iterator['val_data_loader'] = DataLoader(
        valid_data, batch_size=BATCH_SIZE, num_workers=4
    )

    test_data = dataset[dataset_type](read_path=config['test_data_path'])
    data_iterator['test_data_loader'] = DataLoader(
        test_data, batch_size=BATCH_SIZE, num_workers=4
    )

    return data_iterator
