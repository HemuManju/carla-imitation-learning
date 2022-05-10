from datetime import date
from tkinter import E

import pandas as pd
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl


from src.data.stats import classification_accuracy

from src.dataset import warmstart_dataset

from src.architectures.nets import (
    CIRLBasePolicy,
    CIRLFutureLatent,
)

from src.models.imitation import WarmStart
from src.evaluate.agents import CustomCILAgent
from src.evaluate.experiments import CORL2017

from benchmark.run_benchmark import Benchmarking

from src.visualization.visualize import plot_trends

import yaml
from utils import skip_run, get_num_gpus


with skip_run('skip', 'warm_starting') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename='warm_start',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(cfg['logs_path'], name='warm_start')

    # Setup
    net = CIRLBasePolicy(cfg)
    # actions = net(net.example_input_array, net.example_command)
    # print(actions.shape)  # verification

    # Dataloader
    data_loader = warmstart_dataset.webdataset_data_iterator(cfg)
    model = WarmStart(cfg, net, data_loader)
    if cfg['check_point_path'] is None:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=cfg['check_point_path'],
            enable_progress_bar=False,
        )
    trainer.fit(model)

with skip_run('skip', 'warm_starting_with_future_latent') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename='warm_start',
        mode='min',
    )
    logger = pl.loggers.TensorBoardLogger(cfg['logs_path'], name='warm_start')

    # Setup
    # Load the backbone network
    read_path = 'trained_models/1631095225/1631095225_model_epoch_19.pth'
    future_latent_prediction = torch.load(read_path, map_location=torch.device('cpu'))
    cfg['future_latent_prediction'] = future_latent_prediction

    # Testing
    x_in = torch.rand((5, 4, 1, 256, 256)).to('cpu')
    # s_in = torch.rand((5, 4, 4)).to('cpu')
    x_out, x_out_ae, x_out_lat, x_in_lat, s_out = future_latent_prediction(x_in)

    net = CIRLFutureLatent(cfg)
    # actions = net(net.example_input_array, net.example_command)
    # print(actions.shape)  # verification

    # Dataloader
    data_loader = warmstart_dataset.webdataset_data_iterator(cfg)
    model = WarmStart(cfg, net, data_loader)
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
    )
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
        'data/processed/1633690363_log.txt',
    ]
    legends = [
        'Train (Simple)',
        'Validation (Simple)',
        'Train (Lat. att.)',
        'Validation (Lat. att.)',
        'Train (Conv. att.)',
        'Validation (Conv. att.)',
    ]
    plot_trends(paths, legends)

with skip_run('run', 'replay_trained_model') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

    restore_config = {'checkpoint_path': 'logs/2022-05-04/WARMSTART/warm_start.ckpt'}
    model = WarmStart.load_from_checkpoint(
        restore_config['checkpoint_path'],
        hparams=cfg,
        net=CIRLBasePolicy(cfg),
        data_loader=None,
    )

    experiment_cfg = yaml.load(open('configs/experiments.yaml'), Loader=yaml.SafeLoader)
    experiment_suite = CORL2017(experiment_cfg)
    agent = CustomCILAgent(model, cfg)
    benchmark = Benchmarking(cfg, agent, experiment_suite)
    benchmark.run()
