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


class SequentialTorchDataset(Dataset):
    """ A Generalized data loader that handles:
        - semseg and image lodaing
        - sensor data
        - single or a sequence of images
        - frame skipping

    Args:
        Dataset ([hparams]): the hyperparamters
        dataset_type ([string]): train, test, or val
    
    Notes:
    - data stats are: (mean=[0.384, 0.389, 0.403], std=[0.126, 0.123, 0.126]) 

    Returns:
        [tuple of (list) of torch tensors]: input (may contain output) consisting of images,
             sensor data, and semantic labels
        [torch tensor]: output labels
    """
    def __init__(self, hparams, dataset_type='train'):
        self.hparams = hparams
        self.image_files = dict.fromkeys(self.hparams['modalities'])
        self.semantic = 'semantic_label' in self.image_files.keys()
        self.dataset_type = dataset_type


        self.read_path = hparams['data_dir']+ dataset_type + '/'
        print(self.read_path)

        for modal in self.image_files.keys(): # sort by image names
            paths = glob(self.read_path +'*/'+modal+'/*')
            self.image_files[modal] = sorted(paths, key=lambda x: int(x.split(modal+'/')[1].split('.')[0]))

        self.file_idx = [int(name.split('/')[-1].split('.')[0]) for name in self.image_files['camera']]
        self.file_idx = (np.array(self.file_idx) - self.file_idx[0]).tolist()  # always start at 0

        assert len(self.file_idx) == self.file_idx[-1]+1 == len(self.image_files['camera']), '{} {} {}'.format(
           len(self.file_idx), self.file_idx[-1]+1, len(self.image_files['camera']))

        # To combbine the data fromes if havent already done so
        if not os.path.exists(self.read_path + 'sensor_all.json'):
            self._combineSensorDataframes()
        self._load_sensor()

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Normalize(mean=[0.384, 0.389, 0.403], std=[0.126, 0.123, 0.126]),
            # transforms.Grayscale(),
            transforms.ColorJitter(brightness=.5, contrast=.5),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.05, 1.5)),
            transforms.RandomAdjustSharpness(2, p=0.5)
        ]) 

    def _load_sensor(self):
        '''Load the sensor data and labels from a single json file
        '''
        sensor_df = pd.read_json(self.read_path + 'sensor_all.json', orient='index')
        # orient_vals = np.array(sensor_df['orientation'].tolist())

        action_ind = continous_to_discreet(np.array(sensor_df['control'].tolist()))                # autopilot action
        # self.visualize(action_ind)

        redlight_status = np.array(sensor_df['traffic_light'].tolist())                            # redlight status
        redlight_dist = distance_to_discrete(np.array(sensor_df['traffic_light_dist'].tolist()))   # dist to traffic light
        frontcar_dist = distance_to_discrete(np.array(sensor_df['frontcar_dist'].tolist()))        # dist to front car

        # self.visualize(np.array(sensor_df['command'].tolist()))
        sensor = np.vstack([np.array(sensor_df['command'].tolist()),                               # sensor data
                        np.linalg.norm(np.array(sensor_df['velocity'].tolist()), axis=1),
                        # np.linalg.norm(np.array(sensor_df['acceleration'].tolist()), axis=1),
                    ]).T 

        target = np.stack((redlight_status, action_ind, redlight_dist, frontcar_dist), axis=-1)
        
        self.y = target[self.file_idx, None]
        self.sensor_data = sensor[self.file_idx, None]

    def visualize(self, values, discrete=True):
        '''Visualize the data
        '''
        print('Visualizing data - len=', values.shape)
        print('range:', np.min(values), ', ', np.max(values))
        
        if discrete:
            unique, counts = np.unique(values, return_counts=True)
            vals = dict(zip(unique, counts))
            print(vals)
        
        plt.hist(values, density=False, bins='auto', histtype='step')  # density=False would make counts
        plt.ylabel('Values')
        plt.xlabel('Data')
        plt.show()

    def plotSemseg(self, image_array):
        '''Plots the RGB image from the semantic label for visualization
        '''
        # categories: moving obstacles, traffic lights, road markers, road, sidewalk and background
        segclass_colors = {
            0: [70, 70, 70],       # 0: Building 1, Fence 2, Other 3, poles 5, vegetation 9, walls 11
            1: [0, 0, 142],        # 1: pedestrian 4, vehicles 10
            2: [220, 220, 0],      # 2: traffic signs 12, 
            3: [153, 153, 153],    # 3: roadlines 6,
            4: [128,  64, 128],    # 4: road 7
            5: [244, 35, 232],     # 5: sidewalk 8
        }

        semseg_rgb = np.zeros((image_array.shape[-2], image_array.shape[-1], 3))
        for _, (clss, val) in enumerate(segclass_colors.items()):
            semseg_rgb[np.where(image_array==clss)] = val

        return semseg_rgb  # 3 * height * width

    def _combineSensorDataframes(self):
        '''Combines sensor data from different logs into a single json file
        '''
        paths = glob(self.read_path + '*/*.json')
        files = sorted(paths, key=lambda x: int(x.split('sensor_')[1].split('.')[0]))
        dfs = []
        for i in tqdm.tqdm(range(len(files)), desc='Combining sensor data'):
            dfs.append(pd.read_json(files[i], orient='index'))
        finaldf = pd.concat(dfs, ignore_index=False)
        print(finaldf)
        finaldf.to_json(self.read_path + 'sensor_all.json', orient='index')  

    def _processSemseg(self, image_array):
        '''Returns the correct semantic label from image and resizes it
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

        assert len(image_array.shape) == 3
        image_array = image_array[0,:,:]                                            # classes encocded in red channel
        image_array = cv2.resize(image_array, (128,74), cv2.INTER_NEAREST)          # Note: axis are flipped in cv

        semseg_target = np.zeros_like(image_array)
        for _, (clss, val) in enumerate(segclass.items()):
            mask = np.isin(image_array, val)
            semseg_target[np.where(mask)] = clss
        
        return semseg_target  # num_images * height * width

    def _load_file(self, index, modality='camera', mode='single'):
        ''' loads an image file and processes it
        '''
                
        if mode == 'single':
            files = self.image_files[modality][index-1:index] 
            images = imread(files[0])                         # single image
        elif mode == 'multiple':
            files = self.image_files[modality][index - self.hparams['frame_skip']:index]            # no frame skipping
            images = imread_collection(files).concatenate()   # stack of images
        elif mode == 'multiple_frameskip':
            files = self.image_files[modality][index - 3*self.hparams['frame_skip']:index:3]        # Frame skipping (3)
            images = imread_collection(files).concatenate()

        images = np.moveaxis(images, -1,-3)                 # need [ch,height,width] or [num_imgs,ch,height,width]
        if self.hparams['crop_sky']:
            images = images[...,100:,:]                     # if crop_sky


        if modality =='camera':
            images = images/255.0                           # normalize 
            if self.hparams['gray_scale']:                  # convert to grayscale                   
                # TODO: very costly, faster to do this using the the same image loader
                images = np.dot(images[..., :], [0.299, 0.587, 0.114])
            if mode == 'single' and len(images.shape)==2:
                images = np.expand_dims(images, axis=0)   # Only if grayscale and single
            pass
        
        elif modality =='semantic_label':
            if mode =='single':
                images = self._processSemseg(images)                
            else:
                img_list = []
                for i in range(images.shape[0]):
                    img_list.append(self._processSemseg(images[i, ...]))
                images = np.array(img_list)

        # if need [num_imgs*ch, height, width]
        # images = np.reshape(images, (-1, images.shape[-2], images.shape[-1]))
        
        return images

    def _getlabels(self, index, mode='single'):
        '''Get the label data and sensor data
        '''
        ft_step = 1                      # future step for RNN training
        if mode == 'single':
            y = self.y[index].squeeze(0)
            sensor = self.sensor_data[index].squeeze(0)
        elif mode == 'multiple':
            y = self.y[index - self.hparams['frame_skip']:index].squeeze(1)          
            sensor = self.sensor_data[index - self.hparams['frame_skip']:index].squeeze(1) 
        elif mode == 'multiple_frameskip':
            y = self.y[index - 3*self.hparams['frame_skip']:index:3].squeeze(1)
            sensor = self.sensor_data[index - 3*self.hparams['frame_skip']:index:3].squeeze(1)

        return y, sensor

    def __getitem__(self, index):
        if(index < 12): # to handle frame skipping
            index = index+12
    
        # Load the image
        x = self._load_file(index, mode=self.hparams['loading_mode'])
        x = torch.from_numpy(x).type(torch.float32)
        if self.hparams['data_augmentation'] and self.dataset_type=='train':
            x = self.transforms(x)

        x_sem = 0
        if self.semantic: 
            x_sem = self._load_file(index, modality='semantic_label', mode=self.hparams['loading_mode'])
            x_sem = torch.from_numpy(x_sem).type(torch.long).squeeze(0)

        y, sensor = self._getlabels(index, mode=self.hparams['loading_mode'])
        y = torch.from_numpy(y).type(torch.long)
        sensor = torch.from_numpy(sensor).type(torch.float32)
        
        x = (x, sensor, x_sem)
        return x, y

    def __len__(self):
        return len(self.image_files['camera']) - self.hparams['frame_skip']


def sequential_train_val_test_iterator(hparams, modes=['train', 'val', 'test']):
    '''Create train, validation, test datasets
    '''
    data_iterator = {}
    for mode in modes:
        data = SequentialTorchDataset(hparams, dataset_type=mode+'')
        data_iterator[mode+'_dataloader'] = DataLoader(data,
                                                       batch_size=hparams['BATCH_SIZE'],
                                                       shuffle=False, num_workers=hparams['NUM_WORKERS']
                                                       )
    return data_iterator


def continous_to_discreet(y):
    '''Discretize the action data
    '''
    steer = y[:, 0].copy()
    throttle = y[:, 1].copy()
    brake = y[:, 2].copy()
    
    # Discretize
    temp = steer.copy()
    steer[temp>=0.1] = 4.0
    steer[np.logical_and(temp>=0.001, temp<0.1)] = 3.0
    steer[np.logical_and(temp<0.001, temp>-0.001)] = 2.0   # straight steering
    steer[np.logical_and(temp<=-0.001, temp>-0.1)] = 1.0
    steer[temp<=-0.1] = 0.0

    # Discretize throttle and brake
    acc = brake.copy()
    acc[np.logical_and(brake == 1.0, throttle == 0.0)] = 0.0  # when brake=1.0, throttle is always zero
    acc[np.logical_and(throttle>0.0, throttle<=0.25)] = 1.0
    acc[np.logical_and(throttle>0.25, throttle<=0.50)] = 2.0
    acc[throttle>0.75] = 3.0

    cls_factor = max(len(np.unique(acc)), len(np.unique(steer)))
    actions = np.vstack((acc, steer)).T

    # Convert actions to indices: (# max*small_class + large_class)
    if cls_factor == len(np.unique(acc)):
        action_ind = actions[:, 0] + cls_factor * actions[:, 1]
    else:
        action_ind = cls_factor * actions[:, 0] + actions[:, 1]

    return action_ind

def distance_to_discrete(y):
    '''Discretize the distance
    '''
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

def calculateStats(dataloader):
    '''calcualte the statistics of the image data
    '''
    means, stds = [], []
    for i, batch in enumerate(dataloader):
        x, y = batch
        x[0] = x[0].to(torch.device('cuda:0'))
        # x[1] = x[1].to(torch.device('cuda:0'))

        # To calculate mean and std
        means.append(torch.mean(x[0], dim =(0,2,3)).cpu().numpy())
        stds.append(torch.std(x[0], dim =(0,2,3)).cpu().numpy())

    print('means',np.mean(means, axis=0))
    print('stds',np.mean(stds, axis=0))