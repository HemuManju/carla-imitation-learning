import torch
import pytorch_lightning as pl
import splitfolders

from src.dataset import vae_dataset, imitation_dataset

from src.architectures.nets import CNNAutoEncoder, CNNAuxNet, ConvNet1, ConvNetRawSegment, CNNAuxNet

from src.models.model_supervised import Model_Segmentation_Traffic_Light_Supervised

from src.models.vae import VAE
from src.models.imitation import Imitation

from hydra.experimental import compose, initialize
from utils import skip_run, get_num_gpus

# Initialize the config directory
initialize(config_path="configs", job_name="vae")

with skip_run('skip', 'split_image_folder') as check, check():
    hparams = compose(config_name="config")
    hparams['camera'] = 'semantic'
    log = hparams['train_logs'][0]

    read_path = hparams['data_dir'] + 'raw' + '/' + log
    print(read_path)
    splitfolders.ratio(read_path,
                       output=hparams['data_dir'] + 'processed' + '/' + log,
                       seed=1337,
                       ratio=(.8, 0.1, 0.1),
                       shuffle=False)

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

with skip_run('run', 'aux-adv') as check, check():
    # Load the parameters
    hparams = compose(config_name="config", overrides=['model=imitation'])

    camera_type = ['camera']#, 'camera_sFOV']
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

        
        # output = net(net.example_input_array)
        # for it in output:
        #     print(it.shape)

        # print(output)  # verification
        data_loader = imitation_dataset.sequential_train_val_test_iterator(hparams)
        # All this magic number should match the one used when training supervised...
        net = Model_Segmentation_Traffic_Light_Supervised(1, 1, 1024, 6, 4, True, pretrained=True)

        model = Imitation(hparams, net, data_loader)
        trainer = pl.Trainer(gpus=gpus,
                             max_epochs=hparams.NUM_EPOCHS,
                             logger=logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(model)


with skip_run('skip', 'aux') as check, check():
    # Load the parameters
    hparams = compose(config_name="config", overrides=['model=imitation'])

    camera_type = ['camera']#, 'camera_sFOV']
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
        net = CNNAuxNet(hparams)
        ### using pretrained weights
        # ckpt_path='logs/2021-10-29/imitation.ckpt'  # image recons  
        # ckpt_path='logs/2021-11-02/imitation.ckpt'    # semseg
        ckpt_path='logs/2021-12-03/imitation.ckpt'    # semseg
        net.loadWeights(ckpt_path=ckpt_path, selected_subnet=['encoder'])
        net.freezeLayers(selected_subnet=['encoder'])
        
        # output = net(net.example_input_array)
        # print(output)  # verification
        data_loader = imitation_dataset.sequential_train_val_test_iterator(
            hparams)
        model = Imitation(hparams, net, data_loader)
        trainer = pl.Trainer(gpus=gpus,
                             max_epochs=hparams.NUM_EPOCHS,
                             logger=logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(model)


with skip_run('skip', 'test') as check, check():
    # Load the parameters
    hparams = compose(config_name="config", overrides=['model=imitation'])

    camera_type = ['camera']
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
        net = CNNAuxNet(hparams)
        # net = ConvNet1(hparams)
        output = net(net.example_input_array)
        # print(output)  # verification
        data_loader = imitation_dataset.sequential_train_val_test_iterator(
            hparams)
        model = Imitation(hparams, net, data_loader)
        model = model.load_from_checkpoint(
            'logs/2021-12-03/imitation-v1.ckpt', hparams=hparams, net=net, data_loader=data_loader)
        model.calcAccuracy()
        # model.sampleOutput()


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
