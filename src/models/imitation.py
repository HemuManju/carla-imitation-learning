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

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
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
                "monitor": 'val_checkpoint_on',
            },
        }
        return opt

    def scale_image(self, img):
        out = (img + 1) / 2
        return out


class ConditionalImitation(pl.LightningModule):
    def __init__(self, hparams, net, carla_data):
        super(ConditionalImitation, self).__init__()
        self.h_params = hparams
        self.net = net
        self.carla_data = carla_data

        # Save hyper-parameters
        self.save_hyperparameters(self.h_params)

    def forward(self, image, speed):
        branches_out, pred_speed = self.net.forward(image, speed)
        return branches_out, pred_speed

    def calculate_loss(self, branches_out, mask, target, pred_speed, speed):
        # Criterion
        criterion = nn.MSELoss()

        mask_out = branches_out * mask
        branch_loss = criterion(mask_out, target) * 4
        speed_loss = criterion(pred_speed, speed)

        loss = (
            self.h_params['branch_weight'] * branch_loss
            + self.h_params['speed_weight'] * speed_loss
        )
        return loss

    def training_step(self, batch, batch_idx):
        img, speed, target, mask = batch

        # Predict and calculate loss
        branches_out, pred_speed = self.forward(img, speed)
        loss = self.calculate_loss(branches_out, mask, target, pred_speed, speed)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, speed, target, mask = batch

        # Predict and calculate loss
        branches_out, pred_speed = self.forward(img, speed)
        loss = self.calculate_loss(branches_out, mask, target, pred_speed, speed)

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.carla_data.loaders["train"]

    def val_dataloader(self):
        return self.carla_data.loaders["eval"]

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.h_params['LEARNING_RATE'])
        lr_scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.9, verbose=True
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'losses/val_loss',
        }
        return [optimizer]

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

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, command, action = batch[0], batch[1], batch[2]

        # Predict and calculate loss
        output = self.forward(images, command)
        criterion = nn.MSELoss()
        loss = criterion(output, action)

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
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
                "monitor": 'losses/val_loss',
            },
        }
        return opt

    def scale_image(self, img):
        out = (img + 1) / 2
        return out
