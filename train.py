from datetime import date, datetime

import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from src.data.stats import classification_accuracy

from src.dataset.cil_dataset import CarlaH5Data

from src.architectures.nets import CarlaNet


from src.models.imitation import ConditionalImitation
from scenario_runner.run_scenario import RunScenario

from src.visualization.visualize import show_grid, plot_trends

import yaml
from utils import skip_run, get_num_gpus

with skip_run('skip', 'conditional_imitation_learning') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/COND_IMITATION'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'] + f'/logs/time_{datetime.now().strftime("%H_%M_%S")}',
        save_top_k=1,
        filename='{epoch}-{val_loss:.4f}-{train_loss:.4f}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'],
        name='logs',
        version=f'time_{datetime.now().strftime("%H_%M_%S")}',
    )

    # Dataset
    carla_data = CarlaH5Data(
        train_folder=cfg['training_dataset'],
        eval_folder=cfg['validation_dataset'],
        batch_size=cfg['BATCH_SIZE'],
        num_workers=cfg['number_workers'],
    )

    # Neural network
    net = CarlaNet()
    model = ConditionalImitation(cfg, net, carla_data)

    if cfg['check_point_path'] is None:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=cfg['check_point_path'],
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

with skip_run('skip', 'replay_trained_model') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

    restore_config = {'checkpoint_path': 'logs/2022-01-31/WARMSTART/warm_start.ckpt'}

    # Random seed
    torch.manual_seed(cfg['pytorch_seed'])
    net = CIRLBasePolicy(cfg)
    cfg['net'] = net
    cfg['model'] = WarmStart

    RunScenario(cfg=cfg, restore_config=restore_config)
