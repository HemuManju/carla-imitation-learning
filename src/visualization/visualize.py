import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import torchvision.transforms.functional as F


def show_grid(imgs):
    # if not isinstance(imgs, list):
    #     imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach().cpu().numpy()
        # img = F.to_pil_image(img)
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
    return fig


def interactive_show_grid(imgs):
    try:
        fig = show_grid(imgs)
        plt.close(fig)
    except KeyboardInterrupt:
        import sys

        sys.exit()


def plot_trends(paths, legends, config, axs=None):
    dfs = []
    for path in paths:
        dfs.append(pd.read_csv(path))

    if axs is None:
        plt.style.use('clean')
        fig, axs = plt.subplots(figsize=(8, 5))

    for i, df in enumerate(dfs):
        if config['dashes']:
            axs.plot(df['Step'], df['Value'], dashes=config['dashes'])
        else:
            axs.plot(df['Step'], df['Value'])

    plt.xlim([-1, dfs[0]['Step'].values[-1]])
    plt.ylabel('Mean Squared Loss')
    plt.xlabel('Training Steps')


def plot_frames(ax, array):
    try:
        for i in range(array.shape[0]):
            ax[i].imshow(array[i, :, :], origin='lower')
        plt.pause(0.00001)
    except KeyboardInterrupt:
        print("\nshutdown by user")
