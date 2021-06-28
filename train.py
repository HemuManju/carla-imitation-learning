import torch
import pytorch_lightning as pl

from src.data.create_data import compress_data

from src.dataset import vae_dataset, imitation_dataset

from src.architectures.nets import CNNAutoEncoder, ConvNet1

from src.models.vae import VAE
from src.models.imitation import Imitation

from hydra.experimental import compose, initialize
from utils import skip_run, get_num_gpus

# Initialize the config directory
initialize(config_path="configs", job_name="vae")

with skip_run('skip', 'compress_image_data') as check, check():
    # Load the parameters
    hparams = compose(config_name="config")
    compress_data(hparams)

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
    hparams.camera = 'camera'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(hparams.pytorch_seed)

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=hparams.log_dir,
                                                       save_top_k=1,
                                                       filename='imitation',
                                                       mode='min')
    logger = pl.loggers.TensorBoardLogger(hparams.log_dir, name='imitation')

    # Setup
    hparams['train_logs'] = ['Log1']
    net = ConvNet1(hparams)
    actions = net(net.example_input_array)
    print(actions)  # verification
    data_loader = imitation_dataset.train_val_test_iterator(
        hparams, data_split_type='pooled_data')
    model = Imitation(hparams, net, data_loader)
    trainer = pl.Trainer(gpus=gpus,
                         max_epochs=hparams.NUM_EPOCHS,
                         logger=logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(model)
