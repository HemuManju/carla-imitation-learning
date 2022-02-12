import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Imitation(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Imitation, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['train_dataloader']

    def val_dataloader(self):
        return self.data_loader['val_dataloader']

    def test_dataloader(self):
        return self.data_loader['test_dataloader']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.h_params['LEARNING_RATE'])
        opt = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": 'val_loss',
            },
        }
        return opt

    def scale_image(self, img):
        out = (img + 1) / 2
        return out


class WarmStart(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(WarmStart, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

    def forward(self, x, command):
        output = self.net.forward(x, command)
        return output

    def training_step(self, batch, batch_idx):
        images, command, action = batch[0], batch[1], batch[2]

        # Predict and calculate loss
        output = self.forward(images, command)
        criterion = nn.MSELoss()

        loss = criterion(output, action)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, command, action = batch[0], batch[1], batch[2]

        # Predict and calculate loss
        output = self.forward(images, command)
        criterion = nn.MSELoss()
        loss = criterion(output, action)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['training']

    def val_dataloader(self):
        return self.data_loader['validation']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.h_params['LEARNING_RATE'])
        opt = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": 'val_loss',
            },
        }
        return opt

    def scale_image(self, img):
        out = (img + 1) / 2
        return out
