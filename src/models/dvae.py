import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Adam


class DynamicVAE(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(DynamicVAE, self).__init__()
        self.cfg = hparams
        self.net = net
        self.data_loader = data_loader

    def forward(self, x):
        x_out, mu, log_sigma = self.net.forward(x)
        return x_out, mu, log_sigma

    def training_step(self, batch, batch_idx):
        x = batch

        # Encode and decode
        x_out, mu, log_sigma = self.forward(x)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
        )

        # Reconstruction loss
        recon_loss_criterion = nn.MSELoss(reduction='sum')
        recon_loss = recon_loss_criterion(x, x_out)

        # Total loss
        loss = self.cfg['alpha'] * recon_loss + self.cfg['beta'] * kl_loss

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        # Encode and decode
        x_out, mu, log_sigma = self.forward(x)

        # Loss
        kl_loss = -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
        )

        # Reconstruction loss
        recon_loss_criterion = nn.MSELoss(reduction='sum')
        recon_loss = recon_loss_criterion(x, x_out)

        loss = self.cfg['alpha'] * recon_loss + self.cfg['beta'] * kl_loss

        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return x_out, loss

    def train_dataloader(self):
        return self.data_loader['train_data_loader']

    def val_dataloader(self):
        return self.data_loader['val_data_loader']

    def test_dataloader(self):
        return self.data_loader['test_data_loader']

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def scale_image(self, img):
        out = (img + 1) / 2
        return out
