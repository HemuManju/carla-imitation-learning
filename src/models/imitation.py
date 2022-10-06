import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_msssim import MS_SSIM

from .utils import ChamferDistance


class WeightedMSE(torch.nn.MSELoss):
    def __init__(self, weights=None):
        super().__init__(reduction='none')
        self.weights = weights

    def forward(self, input, target):
        if self.weights is not None:
            return torch.mean(
                torch.sum(self.weights * super().forward(input, target), dim=1)
            )
        else:
            return torch.mean(super().forward(input, target))


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Autoencoder, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output, embedding = self.forward(x)
        criterion = MS_SSIM(data_range=1, size_average=True, channel=1)

        loss = 1.0 - criterion(output, y)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output, embedding = self.forward(x)
        criterion = MS_SSIM(data_range=1, size_average=True, channel=1)
        loss = 1.0 - criterion(output, y)

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['training']

    def val_dataloader(self):
        return self.data_loader['validation']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.h_params['LEARNING_RATE'])
        lr_scheduler = ReduceLROnPlateau(
            optimizer, patience=5, factor=0.95, verbose=True
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'losses/val_loss',
        }

        if self.h_params['check_point_path'] is None:
            return [optimizer], [scheduler]
        else:
            return [optimizer]


class Imitation(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Imitation, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        # self.save_hyperparameters(self.h_params)

    def forward(self, x, command):
        output = self.net.forward(x, command)
        return output

    def training_step(self, batch, batch_idx):
        images, command, action = batch[0], batch[1], batch[2]

        # Predict and calculate loss
        output = self.forward(images, command)
        criterion = nn.MSELoss()
        # criterion = WeightedMSE(weights=torch.tensor([1, 1, 1]).to(self.device))
        loss = criterion(output, action)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, command, action = batch[0], batch[1], batch[2]

        # Predict and calculate loss
        output = self.forward(images, command)
        criterion = nn.MSELoss()
        # criterion = WeightedMSE(weights=torch.tensor([1, 1, 1]).to(self.device))
        loss = criterion(output, action)

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['training']

    def val_dataloader(self):
        return self.data_loader['validation']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.h_params['LEARNING_RATE'])
        lr_scheduler = ReduceLROnPlateau(
            optimizer, patience=5, factor=0.95, verbose=True
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'losses/val_loss',
        }

        if self.h_params['check_point_path'] is None:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

