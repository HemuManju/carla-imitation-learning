from torch import nn
import pytorch_lightning as pl
import torch

from .utils import build_model


class ConvNet1(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvNet1, self).__init__()

        # Parameters
        obs_size = hparams['obs_size']
        n_actions = hparams['n_actions']

        self.example_input_array = torch.randn((1, obs_size, 256, 256))

        # Architecture
        self.cnn_base = nn.Sequential(  # input shape (4, 256, 256)
            nn.Conv2d(obs_size, 16, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                nn.Linear(64, 32), nn.ReLU(),
                                nn.Linear(32, n_actions))

    def forward(self, x):
        x = self.cnn_base(x)
        x = torch.flatten(x, start_dim=1)
        q_values = self.fc(x)
        return q_values


class ConvNetRawSegment(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvNet1, self).__init__()

        # Parameters
        obs_size = hparams['obs_size']
        n_actions = hparams['n_actions']

        self.example_input_array = torch.randn((1, obs_size, 256, 256))

        # Architecture
        self.cnn_base = nn.Sequential(  # input shape (4, 256, 256)
            nn.Conv2d(obs_size, 32, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(nn.Linear(256, 200), nn.ReLU(),
                                nn.Linear(200, 48), nn.ReLU(),
                                nn.Linear(48, n_actions))

    def forward(self, x, x_seg):
        out_1 = self.cnn_base(x)
        out_2 = self.cnn_base(x_seg)
        out_1 = torch.flatten(out_1, start_dim=1)
        out_2 = torch.flatten(out_2, start_dim=1)
        x = out_1 + out_2
        q_values = self.fc(x)
        return q_values


class CNNAutoEncoder(pl.LightningModule):
    """
    Simple auto-encoder with MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        self.hidden_size: size of hidden layers
    """
    def __init__(self, hparams, latent_size: int = 128):
        super(CNNAutoEncoder, self).__init__()

        # Parameters
        image_size = hparams['image_size']
        self.example_input_array = torch.randn((2, *image_size))

        self.encoder = build_model(hparams['encoder_config'])
        self.decoder = build_model(hparams['decoder_config'])

        self.fc_mu = nn.Linear(128, latent_size)
        self.fc_log_sigma = nn.Linear(128, latent_size)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        return mu, log_sigma

    def decode(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        reconstructed = self.decoder(x)
        return reconstructed

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        sigma = log_sigma.exp()
        eps = torch.rand_like(sigma)
        z = eps.mul(sigma).add_(mu)
        reconst = self.decode(z)
        return reconst, mu, log_sigma
