from datetime import date

import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from src.data.stats import classification_accuracy

from src.dataset import vae_dataset, imitation_dataset

from src.architectures import layer_config
from src.architectures.nets import CNNAutoEncoder, ConvNet1, ConvNetRawSegment

from src.models.vae import VAE
from src.models.imitation import Imitation

from src.visualization.visualize import show_grid, plot_trends

import yaml
from utils import skip_run, get_num_gpus

with skip_run('skip', 'vae_training') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/vae.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today())

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename='vae',
        mode='min')
    logger = pl.loggers.TensorBoardLogger(cfg['logs_path'], name='vae')

    # Setup the network
    cfg['encoder_config'] = layer_config.layers_encoder_256_128
    cfg['decoder_config'] = layer_config.layers_decoder_256_128
    net = CNNAutoEncoder(cfg)

    # Get the dataloaders
    data_loader = vae_dataset.train_val_test_iterator(cfg)

    # Setup the model and run
    model = VAE(cfg, net, data_loader)
    # Early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        patience=20,
                                        verbose=False,
                                        mode="min")
    trainer = pl.Trainer(gpus=gpus,
                         max_epochs=cfg['NUM_EPOCHS'],
                         logger=logger,
                         callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model)

with skip_run('skip', 'vae_inference') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/vae.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/VAE'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Restore the model and put in eval() mode
    cfg['encoder_config'] = layer_config.layers_encoder_256_128
    cfg['decoder_config'] = layer_config.layers_decoder_256_128
    net = CNNAutoEncoder(cfg)
    check_point_path = 'logs/2021-11-10/vae.ckpt'
    model = VAE.load_from_checkpoint(check_point_path,
                                     hparams=cfg,
                                     net=net,
                                     data_loader=None)
    model = model.to("cuda")
    model.eval()
    model.freeze()

    # Data loader
    data_loader = vae_dataset.train_val_test_iterator(cfg)
    for i, data in enumerate(data_loader['test_data_loader']):
        if i == 40:
            x_out, mu, log_var = model(data.to("cuda"))
            imgs = make_grid(x_out)
            show_grid(imgs)
            imgs = make_grid(data)
            show_grid(imgs)
            plt.show()

with skip_run('skip', 'behavior_cloning') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/vae.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/VAE'

    camera_type = ['camera', 'semantic']
    for camera in camera_type:
        cfg['camera'] = camera

        # Random seed
        gpus = get_num_gpus()
        torch.manual_seed(cfg.pytorch_seed)

        # Checkpoint
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=cfg.log_dir,
            save_top_k=1,
            filename='imitation',
            mode='min')
        logger = pl.loggers.TensorBoardLogger(cfg.log_dir, name='imitation')

        # Setup
        cfg['train_logs'] = ['Log1']
        net = ConvNet1(cfg)
        actions = net(net.example_input_array)
        print(actions)  # verification
        data_loader = imitation_dataset.sequential_train_val_test_iterator(cfg)
        model = Imitation(cfg, net, data_loader)
        trainer = pl.Trainer(gpus=gpus,
                             max_epochs=cfg.NUM_EPOCHS,
                             logger=logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(model)

with skip_run('skip', 'behavior_cloning_with_raw_segmented') as check, check():
    # Load the parameters
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)

    camera_type = ['camera', 'semantic']
    for camera in camera_type:
        cfg['camera'] = camera

        # Random seed
        gpus = get_num_gpus()
        torch.manual_seed(cfg.pytorch_seed)

        # Checkpoint
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=cfg.log_dir,
            save_top_k=1,
            filename='imitation',
            mode='min')
        logger = pl.loggers.TensorBoardLogger(cfg.log_dir, name='imitation')

        # Setup
        cfg['train_logs'] = ['Log1']
        net = ConvNetRawSegment(cfg)
        actions = net(net.example_input_array)
        print(actions)  # verification
        data_loader = imitation_dataset.sequential_train_val_test_iterator(cfg)
        model = Imitation(cfg, net, data_loader)
        trainer = pl.Trainer(gpus=gpus,
                             max_epochs=cfg.NUM_EPOCHS,
                             logger=logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(model)

with skip_run('skip', 'algorithm_stats') as check, check():
    cfg = yaml.load(open('configs/vae.yaml'), Loader=yaml.SafeLoader)
    classification_accuracy(cfg)

with skip_run('skip', 'testing_logics') as check, check():
    data = pd.read_csv('data/external/accs.csv', na_values=0).fillna(0)
    df = data.max(axis=1)
    print(df.mean() * 100, df.std() * 100)

with skip_run('skip', 'figure_plotting') as check, check():
    plt.style.use('clean')
    plt.rcParams['axes.grid'] = True
    paths = [
        'data/processed/1632282330_log.txt',
        'data/processed/1633690232_log.txt',
        'data/processed/1633690363_log.txt'
    ]
    legends = [
        'Train (Simple)', 'Validation (Simple)', 'Train (Lat. att.)',
        'Validation (Lat. att.)', 'Train (Conv. att.)',
        'Validation (Conv. att.)'
    ]
    plot_trends(paths, legends)
