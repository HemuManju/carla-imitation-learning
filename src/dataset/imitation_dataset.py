from itertools import count
import os
from glob import glob

from matplotlib import pyplot as plt
from numpy.lib.type_check import imag
from skimage.io import imread_collection, imread
from PIL import Image, ImageOps
from time import time
import pandas as pd
import tqdm
import cv2

from sklearn.model_selection import train_test_split

import torch
import numpy as np
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


class LargeTorchDataset(Dataset):
    def __init__(self, hparams, dataset_type='train'):
        # Read path
        log = hparams['train_logs'][0]
        self.read_path = hparams[
            'data_dir'] + 'processed' + '/' + log + '/' + dataset_type + '/' + hparams[
                'camera']
        self.image_files = os.listdir(self.read_path)

        # Get corresponding targets (autopilot actions)
        self.file_idx = [
            int(name.split('.')[0]) - 1 for name in self.image_files
        ]  # file name starts from 1
        autopilot_actions = np.genfromtxt(hparams['data_dir'] + 'raw' + '/' +
                                          log + '/state.csv',
                                          delimiter=',',
                                          usecols=(4, 5, 6, 7))
        action_ind = continous_to_discreet(autopilot_actions)
        actions = np.stack(action_ind, axis=-1)
        self.y = actions[self.file_idx, None]

        # This step normalizes image between 0 and 1
        self.transform = transforms.ToTensor()

    def _load_file(self, file_name):
        image = imread(self.read_path + '/' + file_name, as_gray=True)
        return image

    def __getitem__(self, index):
        # Load
        x = self._load_file(self.image_files[index])

        # Transform
        x = self.transform(x).type(torch.float32)
        y = torch.from_numpy(self.y[index]).type(torch.long)
        return x, y.squeeze(-1)

    def __len__(self):
        return len(self.image_files)

class SequentialTorchDataset(Dataset):
    def __init__(self, hparams, dataset_type='train'):
        self.hparams = hparams
        self.read_path = hparams['data_dir']+ dataset_type + '/'
        print(self.read_path)
        self.image_files = dict.fromkeys(['camera', 'semantic_label']) 
        self.semantic = 'semantic_label' in self.image_files.keys()

        for modal in self.image_files.keys():
            self.image_files[modal] = sorted(glob(self.read_path +'*/'+ modal + '/*'))

        self.file_idx = [int(name.split('/')[-1].split('.')[0]) for name in self.image_files['camera']]
        self.file_idx = (np.array(self.file_idx) - self.file_idx[0]).tolist()  # always start at 0

        assert len(self.file_idx) == self.file_idx[-1]+1 == len(self.image_files['camera'])

        # self._combineSensorDataframes()
        self._load_sensor()

    def _load_sensor(self):
        '''Load the data from a single json file
        '''
        sensor_df = pd.read_json(self.read_path + 'sensor_all.json', orient='index')
        # orient_vals = np.array(sensor_df['orientation'].tolist())

        action_ind = continous_to_discreet(np.array(sensor_df['control'].tolist()))                # autopilot action
        # self._visualize(action_ind)

        redlight_status = np.array(sensor_df['traffic_light'].tolist())                            # redlight status
        redlight_dist = distance_to_discrete(np.array(sensor_df['traffic_light_dist'].tolist()))   # dist to traffic light
        frontcar_dist = distance_to_discrete(np.array(sensor_df['frontcar_dist'].tolist()))        # dist to front car

        sensor = np.vstack([np.array(sensor_df['command'].tolist()),                               # sensor data
                        np.linalg.norm(np.array(sensor_df['velocity'].tolist()), axis=1),
                        np.linalg.norm(np.array(sensor_df['acceleration'].tolist()), axis=1)
                    ]).T

        target = np.stack((redlight_status, action_ind, redlight_dist, frontcar_dist), axis=-1)
        
        self.y = target[self.file_idx, None]
        self.sensor_data = sensor[self.file_idx, None]

    def _visualize(self, values):
        print('Visualizing data - len=', values.shape)
        print('range:', np.min(values), ', ', np.max(values))
        # pass
        unique, counts = np.unique(values, return_counts=True)
        vals = dict(zip(unique, counts))
        print(vals)
        # counts = dict((i, values.count(i)) for i in values)
        
        plt.hist(values, density=False, bins='auto', histtype='step')  # density=False would make counts
        plt.ylabel('Values')
        plt.xlabel('Data')
        plt.show()

    def _combineSensorDataframes(self):
        '''Combines sensor data from different logs into a single json file
        '''
        files = sorted(glob( self.read_path + '*/*.json'))
        dfs = []
        for i in tqdm.tqdm(range(len(files)), desc='Combining sensor data'):
            dfs.append(pd.read_json(files[i], orient='index'))
        finaldf = pd.concat(dfs, ignore_index=False)
        print(finaldf)
        finaldf.to_json(self.read_path + 'sensor_all.json', orient='index')  

    def _processSemseg(self, image_array):
        '''Returns the correct semantic label from image
        '''
        # categories: moving obstacles, traffic lights, road markers, road, sidewalk and background
        segclass = {
            0: [1,2,3,5,9,11],         # 0: Building 1, Fence 2, Other 3, poles 5, vegetation 9, walls 11
            1: [4, 10],                # 1: pedestrian 4, vehicles 10
            2: [12],                   # 2: traffic signs 12, 
            3: [6],                    # 3: roadlines 6,
            4: [7],                    # 4: road 7
            5: [8],                    # 5: sidewalk 8
        }

        # TODO: make this more efficient
        # resize the image
        # img = torch.from_numpy(image_array)
        # img = self.transform_sem(img)
        # img = img.numpy()
        # temp = img[0,:,:]

        image_array = image_array[0,:,:]                                    # classes encocded in red channel
        image_array = cv2.resize(image_array, (128,74), cv2.INTER_NEAREST)  # Note: axis are flipped in cv

        semseg_target = np.zeros_like(image_array)
        for clss, val in enumerate(segclass):
            mask = np.isin(image_array, val)
            semseg_target[np.where(mask)] = clss
        
        return semseg_target  # num_images * height * width

    def _load_file(self, index, modality='camera', mode='single'):

        if mode =='single':
            files = self.image_files[modality][index-1:index] 
            images = imread(files[0])                         # single image
        elif mode=='multiple':
            files = self.image_files[modality][index - self.hparams['frame_skip']:index]            # no frame skipping
            images = imread_collection(files).concatenate()   # stack of images
        elif mode=='multiple_frameskip':
            files = self.image_files[modality][index - 3*self.hparams['frame_skip']:index:3]        # Frame skipping (3)
            images = imread_collection(files).concatenate()

        images = np.moveaxis(images, -1,-3)         # need [ch,height,width] or [num_imgs,ch,height,width]
        images = images[...,120:,:]                 # if crop_sky

        if modality =='camera':                               
            # TODO: very costly, faster to do this using the the same image loader
            # images = np.dot(images[..., :], [0.299, 0.587, 0.114]) / 255.0    # convert to grayscale
            if mode == 'single' and len(images.shape)==2:
                pass
                # images = np.expand_dims(images, axis=0)   # Only if grayscale and single
            pass
        
        elif modality =='semantic_label':
            if mode is 'single':
                images = self._processSemseg(images)                
            else:
                img_list = []
                for i in range(images.shape[0]):
                    img_list.append(self._processSemseg(images[i, ...]))
                images = np.array(img_list)

        images = np.reshape(images, (-1, images.shape[-2], images.shape[-1]))  # [num_imgs*ch, height, width]
        
        return images

    def __getitem__(self, index):
        if(index < 12): # to handle frame skipping
            index = index+12
        
        x_sem = None
        # Load the image
        x = self._load_file(index, mode='single')
        x = torch.from_numpy(x).type(torch.float32)

        if self.semantic: 
            x_sem = self._load_file(index, modality='semantic_label', mode='single')
            x_sem = torch.from_numpy(x_sem).type(torch.long).squeeze(0)

        y = torch.from_numpy(self.y[index]).type(torch.long).squeeze(0)
        sensor = torch.from_numpy(self.sensor_data[index]).type(torch.float32).squeeze(0)
        
        x = (x, sensor, x_sem)
        return x, y

    def __len__(self):
        return len(self.image_files['camera']) - self.hparams['frame_skip']


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


def large_train_val_test_iterator(hparams):
    # Parameters
    BATCH_SIZE = hparams['BATCH_SIZE']

    # Create train, validation, test datasets
    data_iterator = {}
    train_data = LargeTorchDataset(hparams, dataset_type='train')
    data_iterator['train_dataloader'] = DataLoader(train_data,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)

    valid_data = LargeTorchDataset(hparams, dataset_type='val')
    data_iterator['val_dataloader'] = DataLoader(valid_data,
                                                 batch_size=BATCH_SIZE)

    test_data = LargeTorchDataset(hparams, dataset_type='test')
    data_iterator['test_dataloader'] = DataLoader(test_data,
                                                  batch_size=BATCH_SIZE)

    return data_iterator


def sequential_train_val_test_iterator(hparams):
    # Parameters
    BATCH_SIZE = hparams['BATCH_SIZE']
    NUM_WORKERS = hparams['NUM_WORKERS']

    # Create train, validation, test datasets
    data_iterator = {}
    train_data = SequentialTorchDataset(hparams, dataset_type='train')
    data_iterator['train_dataloader'] = DataLoader(train_data,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False, num_workers=NUM_WORKERS
                                                   )

    valid_data = SequentialTorchDataset(hparams, dataset_type='val')
    data_iterator['val_dataloader'] = DataLoader(valid_data,
                                                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    test_data = SequentialTorchDataset(hparams, dataset_type='test')
    data_iterator['test_dataloader'] = DataLoader(test_data,
                                                  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    return data_iterator


def continous_to_discreet(y):
    steer = y[:, 0].copy()
    throttle = y[:, 1].copy()
    brake = y[:, 2].copy()
    
    # Discretize
    temp = steer.copy()
    steer[temp>0.001] = 2.0
    steer[temp<-0.001] = 0.0
    steer[~np.logical_or(temp>0.001, temp<-0.001)] = 1.0

    # Discretize throttle and brake
    acc = brake.copy()
    acc[np.logical_and(brake == 1.0, throttle == 0.0)] = 0.0  # when brake=1.0, throttle is always zero
    acc[np.logical_and(throttle>0.0, throttle<=0.25)] = 1.0
    acc[np.logical_and(throttle>0.25, throttle<=0.50)] = 2.0
    acc[throttle>0.75] = 3.0

    actions = np.vstack((acc, steer)).T
    # Convert actions to indices
    action_ind = actions[:, 0] * len(np.unique(acc)) + actions[:, 1]

    return action_ind

def distance_to_discrete(y):
    dist_ind = y.copy()
    dist_ind[y<=10.0] = 0
    dist_ind[np.logical_and(y>10, y<=15)] = 1
    dist_ind[np.logical_and(y>15, y<=20)] = 2
    dist_ind[y>20] = 3

    return dist_ind

def clb_weight(y):
    '''calculate clb weight values
    '''
    ## class-balanced loss weight
    unique, counts = np.unique(y, return_counts=True)
    beta = 0.99999#0.998     # CB loss (b=0 no weighing, b=1 means weighing by inverse frequency)
    class_weight = (1.0-beta)/(1.0-np.power(beta, counts))
    print(counts, 1.0/counts)
    print('class_weight',class_weight)

    # class_weight = np.reciprocal(traindata_sum, where = traindata_sum > 0)
    class_weight = torch.from_numpy(class_weight)

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
