import os

from subprocess import Popen
import atexit

import numpy as np
import deepdish as dd
from skimage.io import imread_collection


def start_shell_command_and_wait(command):
    p = Popen(command, shell=True, preexec_fn=os.setsid)

    def cleanup():
        os.killpg(os.getpgid(p.pid), 15)

    atexit.register(cleanup)
    p.wait()
    atexit.unregister(cleanup)


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


def download_CORL2017_dataset():
    google_drive_download_id = "1hloAeyamYn-H6MfV1dRtY1gJPhkR55sY"
    filename_to_save = "./CORL2017ImitationLearningData.tar.gz"
    download_command = (
        'wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm='
        '$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies '
        '--no-check-certificate \"https://docs.google.com/uc?export=download&id={}\" -O- | '
        'sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id={}" -O {} && rm -rf /tmp/cookies.txt'.format(
            google_drive_download_id, google_drive_download_id, filename_to_save
        )
    )

    print(download_command)

    # start downloading and wait for it to finish
    start_shell_command_and_wait(download_command)
