import torch
import pytorch_lightning as pl
import splitfolders

from src.data.stats import classification_accuracy

from src.dataset import vae_dataset, imitation_dataset

from src.architectures.nets import CNNAutoEncoder, ConvNet1, ConvNetRawSegment

from src.models.vae import VAE
from src.models.imitation import Imitation

from hydra.experimental import compose, initialize
from utils import skip_run, get_num_gpus

# Initialize the config directory
initialize(config_path="configs", job_name="vae")

with skip_run('skip', 'split_image_folder') as check, check():
    hparams = compose(config_name="config")
    hparams['camera'] = 'camera'
    log = hparams['train_logs'][0]

    read_path = hparams['data_dir'] + 'raw' + '/' + log
    print(read_path)
    splitfolders.ratio(read_path,
                       output=hparams['data_dir'] + 'processed' + '/' + log,
                       seed=1337,
                       ratio=(.8, 0.1, 0.1))

with skip_run('skip', 'pooled_data_vae') as check, check():
    # Load the parameters
    hparams = compose(config_name="config")
    hparams.camera = 'SL'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(hparams.pytorch_seed)

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=hparams.log_dir,
                                                       save_top_k=1,
                                                       filename='vae',
                                                       mode='min')
    logger = pl.loggers.TensorBoardLogger(hparams.log_dir, name='vae')

    # Setup
    hparams['train_logs'] = ['Log1', 'Log2', 'Log3', 'Log4', 'Log5', 'Log6']
    net = CNNAutoEncoder(hparams)
    x_out, mu, log_var = net(net.example_input_array)
    data_loader = vae_dataset.train_val_test_iterator(
        hparams, data_split_type='pooled_data')
    model = VAE(hparams, net, data_loader)
    trainer = pl.Trainer(gpus=gpus,
                         max_epochs=hparams.NUM_EPOCHS,
                         logger=logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(model)

with skip_run('skip', 'leave_one_out_data_vae') as check, check():
    # Load the parameters
    hparams = compose(config_name="config")
    hparams.logs
    hparams.camera = 'SL'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(hparams.pytorch_seed)

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=hparams.log_dir,
                                                       save_top_k=1,
                                                       filename='vae',
                                                       mode='min')
    logger = pl.loggers.TensorBoardLogger(hparams.log_dir, name='vae')

    # Setup
    hparams['train_logs'] = ['Log1', 'Log2', 'Log3', 'Log4', 'Log5']
    hparams['test_logs'] = ['Log6']
    net = CNNAutoEncoder(hparams)
    x_out, mu, log_var = net(net.example_input_array)
    data_loader = vae_dataset.train_val_test_iterator(
        hparams, data_split_type='leave_one_out_data')
    model = VAE(hparams, net, data_loader)
    trainer = pl.Trainer(gpus=gpus,
                         max_epochs=hparams.NUM_EPOCHS,
                         logger=logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(model)

with skip_run('skip', 'behavior_cloning') as check, check():
    # Load the parameters
    hparams = compose(config_name="config", overrides=['model=imitation'])

    camera_type = ['camera', 'semantic']
    for camera in camera_type:
        hparams['camera'] = camera

        # Random seed
        gpus = get_num_gpus()
        torch.manual_seed(hparams.pytorch_seed)

        # Checkpoint
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=hparams.log_dir,
            save_top_k=1,
            filename='imitation',
            mode='min')
        logger = pl.loggers.TensorBoardLogger(hparams.log_dir,
                                              name='imitation')

        # Setup
        hparams['train_logs'] = ['Log1']
        net = ConvNet1(hparams)
        actions = net(net.example_input_array)
        print(actions)  # verification
        data_loader = imitation_dataset.sequential_train_val_test_iterator(
            hparams)
        model = Imitation(hparams, net, data_loader)
        trainer = pl.Trainer(gpus=gpus,
                             max_epochs=hparams.NUM_EPOCHS,
                             logger=logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(model)

with skip_run('skip', 'behavior_cloning_with_raw_segmented') as check, check():
    # Load the parameters
    hparams = compose(config_name="config", overrides=['model=imitation'])

    camera_type = ['camera', 'semantic']
    for camera in camera_type:
        hparams['camera'] = camera

        # Random seed
        gpus = get_num_gpus()
        torch.manual_seed(hparams.pytorch_seed)

        # Checkpoint
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=hparams.log_dir,
            save_top_k=1,
            filename='imitation',
            mode='min')
        logger = pl.loggers.TensorBoardLogger(hparams.log_dir,
                                              name='imitation')

        # Setup
        hparams['train_logs'] = ['Log1']
        net = ConvNetRawSegment(hparams)
        actions = net(net.example_input_array)
        print(actions)  # verification
        data_loader = imitation_dataset.sequential_train_val_test_iterator(
            hparams)
        model = Imitation(hparams, net, data_loader)
        trainer = pl.Trainer(gpus=gpus,
                             max_epochs=hparams.NUM_EPOCHS,
                             logger=logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(model)

with skip_run('skip', 'algorithm_stats') as check, check():
    hparams = compose(config_name="config")
    classification_accuracy(hparams)
