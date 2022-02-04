import torch
import pytorch_lightning as pl
import splitfolders

from src.dataset import vae_dataset, imitation_dataset

from src.architectures.nets import CNNAutoEncoder, CNNAuxNet, ConvNet1, ConvNetRawSegment, CNNAuxNet

from src.models.model_supervised import Model_Segmentation_Traffic_Light_Supervised
from src.models.model_supervised_rnn import Model_Segmentation_Traffic_Light_Supervised_RNN

from src.models.vae import VAE
from src.models.imitation import Imitation

from hydra.experimental import compose, initialize
from utils import skip_run, get_num_gpus

# Initialize the config directory
initialize(config_path="configs", job_name="vae")


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
    ckpt_paths = ['logs/all_aux_supervised/imitation-epoch=11-val_loss=0.58-train_loss=0.42.ckpt']          # encoder
    ckpt_paths = ['logs/supervised_rnn_all_aux_latentonly_3frameskip/imitation-epoch=20-val_loss=0.12-train_loss=0.05.ckpt']  # rnn
    
    for ckpt_path in ckpt_paths:

        # Random seed
        gpus = get_num_gpus()
        torch.manual_seed(hparams.pytorch_seed)

        # Checkpoint
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=hparams.log_dir,
            # period=2,
            save_top_k=-1,
            filename='imitation-{epoch}-{val_loss:.2f}-{train_loss:.2f}',
            mode='min')
        logger = pl.loggers.TensorBoardLogger(hparams.log_dir,
                                              name='imitation')

        ### create the AE net
        net_ae = Model_Segmentation_Traffic_Light_Supervised(hparams)
        
        # if want to load weights
        selected_subnets = ['fc_action']
        # net_ae.loadWeights(ckpt_path=ckpt_path, selected_subnet=selected_subnets, exclude_mode=True)
        # net_ae.freezeLayers(selected_subnet=selected_subnets, exclude_mode=True)
        # net = net_ae

        ### create the RNN net from AE
        net = Model_Segmentation_Traffic_Light_Supervised_RNN(hparams, net_ae)
        net.loadWeights(ckpt_path=ckpt_path, selected_subnet=selected_subnets, exclude_mode=True)
        net.freezeLayers(selected_subnet=selected_subnets, exclude_mode=True)
        
        
        # output = net(net.example_input_array)
        # print(output)  # verification
        data_loader = imitation_dataset.sequential_train_val_test_iterator(hparams, modes=['train', 'val'])

        model = Imitation(hparams, net, data_loader)
        trainer = pl.Trainer(gpus=gpus,
                             max_epochs=hparams.NUM_EPOCHS,
                             logger=logger,
                             callbacks=[checkpoint_callback])
        
        ### Log the Loss before training
        trainer.validate(model, dataloaders=[data_loader['val_dataloader']])
        model.val_loss_in_valStep = False
        trainer.validate(model, dataloaders=[data_loader['train_dataloader']])
        model.val_loss_in_valStep = True
        
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
        # net = CNNAuxNet(hparams)
    
        ### using pretrained weights
        ckpt_path='logs/2021-12-12/imitation.ckpt'    # adv
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

    ckpt_paths = ['logs/supervised_image_recons_only/imitation-epoch=15-val_loss=0.05-train_loss=0.05.ckpt']
    ckpt_paths = ['logs/2022-01-17/imitation-epoch=9-val_loss=0.38-train_loss=0.16.ckpt']
    
    for ckpt_path in ckpt_paths:
        # Random seed
        torch.manual_seed(hparams.pytorch_seed)

        # Setup
        net = Model_Segmentation_Traffic_Light_Supervised(hparams)
        # output = net(net.example_input_array)
        # print(output)  # verification

        data_loader = imitation_dataset.sequential_train_val_test_iterator(
            hparams, modes=[ 'val'])

        model = Imitation(hparams, net, data_loader)
        ckpt = ckpt_path
        model = model.load_from_checkpoint(ckpt, hparams=hparams, net=net, data_loader=data_loader)
        
        model.calcAccuracy(dataset_type='val')
        # model.sampleOutput(dataset_type='test')