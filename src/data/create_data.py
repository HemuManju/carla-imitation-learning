import numpy as np
import deepdish as dd
from skimage.io import imread_collection


def compress_data(config):
    data = {}
    # for log in config['logs'][0]:
    log = 'Log1'
    # for camera in config['camera'][0]:
    camera = 'FL'
    read_path = (
        config['data_dir']
        + 'raw'
        + '/'
        + log
        + '/'
        + camera
        + '_resized_224_bw'
        + '/*.png'
    )
    temp_data = imread_collection(read_path)
    all_images = [image[np.newaxis, ...] for image in temp_data]
    data['test'] = np.concatenate(all_images, dtype=np.int8)
    dd.io.save('test.h5', data)
    return None
