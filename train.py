import os
from datetime import date
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import webdataset as wbs

import torch
import pytorch_lightning as pl

from benchmark.core.carla_core import CarlaCore
from benchmark.core.carla_core import kill_all_servers

from src.dataset import imitation_dataset, rnn_dataset
from src.dataset.utils import WebDatasetReader

from src.architectures.nets import (
    CARNet,
    CNNAutoEncoder,
    CIRLCARNet,
    CIRLBasePolicy,
    CIRLRegressorPolicy,
)


from src.models.imitation import Imitation
from src.models.utils import load_checkpoint, number_parameters
from src.evaluate.agents import PIDCILAgent
from src.evaluate.experiments import CORL2017

from benchmark.run_benchmark import Benchmarking
from benchmark.summary import summarize


import yaml
from utils import skip_run, get_num_gpus

with skip_run('skip', 'imitation_with_basenet_gru') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'imitation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'imitation_{navigation_type}'
    )

    # Load the network
    net = CIRLRegressorPolicy(cfg)
    # output = net(net.example_input_array, net.example_command)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    if cfg['check_point_path'] is None:
        model = Imitation(cfg, net, data_loader)
    else:
        model = Imitation.load_from_checkpoint(
            cfg['check_point_path'], hparams=cfg, net=net, data_loader=data_loader,
        )
    # Trainer
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )

    trainer.fit(model)

with skip_run('skip', 'basenet_gru_validation') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Load the network
    restore_config = {
        'checkpoint_path': f'logs/2022-10-06/IMITATION/imitation_{navigation_type}.ckpt'
    }
    model = Imitation.load_from_checkpoint(
        restore_config['checkpoint_path'],
        hparams=cfg,
        net=CIRLRegressorPolicy(cfg),
        data_loader=None,
    )
    model.eval()

    # Load the dataloader
    dataset = imitation_dataset.webdataset_data_test_iterator(
        cfg,
        file_path=f'/home/hemanth/Desktop/carla_data/Town01_NAVIGATION/{navigation_type}/Town01_HardRainNoon_cautious_000007.tar',
    )

    predicted_waypoints = []
    true_waypoints = []
    test = []
    for i, data in enumerate(dataset):
        images, commands, actions = data[0], data[1], data[2]
        out = model(images.unsqueeze(0), torch.tensor(commands).unsqueeze(0))
        actions = actions.reshape(-1, 2).detach().numpy()
        out = out.reshape(-1, 2).detach().numpy()

        # Waypoints from the data
        test.append(data[3]['waypoints'])

        # Project to the world
        predicted = imitation_dataset.project_to_world_frame(out, data[3])
        predicted_waypoints.append(predicted)

        groud_truth = imitation_dataset.project_to_world_frame(actions, data[3])
        true_waypoints.append(groud_truth)

        if i > 1000:
            break

    true_waypoints = np.vstack(true_waypoints)
    plt.scatter(true_waypoints[:, 0], true_waypoints[:, 1])

    # test = np.array(sum(test, []))
    # plt.scatter(test[:, 0], test[:, 1], s=10, marker='s')

    pred_waypoints = np.vstack(predicted_waypoints)
    plt.scatter(pred_waypoints[:, 0], pred_waypoints[:, 1])
    plt.show()

with skip_run('skip', 'dataset_analysis') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Dataset reader
    reader = WebDatasetReader(
        cfg,
        file_path=f'/home/hemanth/Desktop/carla_data_new/Town01_NAVIGATION/{navigation_type}/Town01_HardRainNoon_cautious_000002.tar',
    )
    dataset = reader.get_dataset(concat_n_samples=1)
    waypoint_data = []
    location = []
    reprojected = []
    direction = []

    for i, data in enumerate(dataset):
        data = data['json'][0]
        waypoints = data['waypoints']
        direction.append(np.array(data['moving_direction']))
        projected_ego = imitation_dataset.project_to_ego_frame(data)
        projected_world = imitation_dataset.project_to_world_frame(projected_ego, data)
        reprojected.append(projected_world)
        waypoint_data.append(waypoints)
        location.append(data['location'])
        if i > 1000:
            break

    test_way = np.array(sum(waypoint_data, []))
    directions = np.array(direction)
    test_loc = np.array(location)
    reproj_test = np.concatenate(reprojected)
    plt.quiver(
        test_loc[:, 0],
        test_loc[:, 1],
        directions[:, 0],
        directions[:, 1],
        linewidths=10,
    )
    plt.scatter(test_way[:, 0], test_way[:, 1])
    # plt.scatter(test_loc[:, 0], test_loc[:, 1], marker='s')
    plt.scatter(reproj_test[:, 0], reproj_test[:, 1], s=10, marker='s')
    plt.show()

with skip_run('skip', 'imitation_with_carnet') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'imitation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'imitation_{navigation_type}'
    )

    # Setup
    # Load the backbone network
    read_path = f'logs/2022-07-07/IMITATION/imitation_{navigation_type}.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNet(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['carnet'] = carnet

    # Testing
    # reconstructed, rnn_embeddings = carnet(carnet.example_input_array)

    net = CIRLCARNet(cfg)
    # net(net.example_input_array, net.example_command)

    # net(net.example_input_array)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    model = Imitation(cfg, net, data_loader)
    if cfg['check_point_path'] is None:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
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

with skip_run('skip', 'benchmark_gru_model') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)

    # Experiment_config and experiment suite
    experiment_cfg = yaml.load(open('configs/experiments.yaml'), Loader=yaml.SafeLoader)
    experiment_suite = CORL2017(experiment_cfg)

    # Carla server
    # Setup carla core and experiment
    kill_all_servers()
    os.environ["CARLA_ROOT"] = cfg['carla_server']['carla_path']
    core = CarlaCore(cfg['carla_server'])

    # Get all the experiment configs
    all_experiment_configs = experiment_suite.get_experiment_configs()
    for exp_id, config in enumerate(all_experiment_configs):
        # Update the summary writer info
        town = config['town']
        navigation_type = config['navigation_type']
        weather = config['weather']
        config['summary_writer']['directory'] = f'{town}_{navigation_type}_{weather}'

        # Update the model
        restore_config = {
            'checkpoint_path': f'logs/2022-10-06/IMITATION//imitation_{navigation_type}.ckpt'
        }

        model = Imitation.load_from_checkpoint(
            restore_config['checkpoint_path'],
            hparams=cfg,
            net=CIRLRegressorPolicy(cfg),
            data_loader=None,
        )

        # Change agent
        agent = PIDCILAgent(model=model, config=cfg)

        # Setup the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)

        # Run the benchmark
        benchmark.run(config, exp_id)

    # Kill all servers
    kill_all_servers()(model, cfg)

with skip_run('skip', 'benchmark_trained_carnet_model') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)

    # Experiment_config and experiment suite
    experiment_cfg = yaml.load(open('configs/experiments.yaml'), Loader=yaml.SafeLoader)
    experiment_suite = CORL2017(experiment_cfg)

    # Carla server
    # Setup carla core and experiment
    kill_all_servers()
    os.environ["CARLA_ROOT"] = cfg['carla_server']['carla_path']
    core = CarlaCore(cfg['carla_server'])

    # Get all the experiment configs
    all_experiment_configs = experiment_suite.get_experiment_configs()
    for exp_id, config in enumerate(all_experiment_configs):

        # Update the summary writer info
        town = config['town']
        navigation_type = config['navigation_type']
        weather = config['weather']
        config['summary_writer']['directory'] = f'{town}_{navigation_type}_{weather}'

        # Update the model
        restore_config = {
            'checkpoint_path': f'logs/2022-08-25/WARMSTART/{navigation_type}_last.ckpt'
        }

        # Setup
        # Load the backbone network
        read_path = f'logs/2022-07-07/IMITATION/imitation_{navigation_type}.ckpt'
        cnn_autoencoder = CNNAutoEncoder(cfg)
        carnet = CARNet(cfg, cnn_autoencoder)
        carnet = load_checkpoint(carnet, checkpoint_path=read_path)
        cfg['carnet'] = carnet

        model = Imitation.load_from_checkpoint(
            restore_config['checkpoint_path'],
            hparams=cfg,
            net=CIRLCARNet(cfg),
            data_loader=None,
        )

        # Change agent
        agent = CILAgent(model, cfg)
        # agent = PIDCILAgent(model, cfg)
        # agent = PIDCILAgent(model, cfg)

        # Run the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)
        benchmark.run(config, exp_id)

    # Kill all servers
    kill_all_servers()

with skip_run('skip', 'summarize_benchmark') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

    # towns = ['Town02', 'Town01']
    # weathers = ['ClearSunset', 'SoftRainNoon']
    # navigation_types = ['straight', 'one_curve', 'navigation']

    towns = ['Town01']
    weathers = ['SoftRainNoon']  #'ClearSunset',
    navigation_types = ['navigation']

    for town, weather, navigation_type in itertools.product(
        towns, weathers, navigation_types
    ):
        path = f'logs/benchmark_results/{town}_{navigation_type}_{weather}/measurements.csv'
        print('-' * 32)
        print(town, weather, navigation_type)
        summarize(path)

with skip_run('skip', 'benchmark_trained_model') as check, check():
    # Load the configuration
    navigation_type = 'one_curve'

    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)

    raw_data_path = cfg['raw_data_path']
    cfg['raw_data_path'] = raw_data_path + f'/{navigation_type}'

    restore_config = {'checkpoint_path': 'logs/2022-06-06/one_curve/warmstart.ckpt'}
    model = Imitation.load_from_checkpoint(
        restore_config['checkpoint_path'],
        hparams=cfg,
        net=CARNet(cfg),
        data_loader=None,
    )
    model.freeze()
    model.eval()
    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Dataloader
    data_loader = rnn_dataset.webdataset_data_iterator(cfg)

    for data in data_loader['training']:
        output = model(data[0][0:1], data[1][0:1])
        print(data[2][0:1])
        # print(torch.max(data[2][:, 0] / 20))
        print(output)
        print('-------------------')
