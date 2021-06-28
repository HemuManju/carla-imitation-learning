from torch import nn
import pytorch_lightning as pl
import torch


class ConvNet1(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvNet1, self).__init__()

        # Parameters
        obs_size = hparams['obs_size']
        n_actions = hparams['n_actions']

        self.example_input_array = torch.randn((1, 1, 256, 256))

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

    def forward(self, x):
        x = self.cnn_base(x)
        x = torch.flatten(x, start_dim=1)
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
    def __init__(self, hparams, z_size: int = 32):
        super(CNNAutoEncoder, self).__init__()

        # Parameters
        image_size = hparams.image_size
        self.example_input_array = torch.randn((1, *image_size))

        self.encoder = nn.Sequential(
            nn.Conv2d(image_size[0], 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=6, stride=3), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=6, stride=3), nn.ReLU())

        self.hidden_size = self._get_flatten_size()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, 128, kernel_size=6, stride=2),
            nn.ReLU(), nn.ConvTranspose2d(128, 128, kernel_size=6, stride=2),
            nn.ReLU(), nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2),
            nn.ReLU(), nn.ConvTranspose2d(64, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_size[0], kernel_size=4, stride=2),
            nn.Sigmoid())
        self.to_mu = nn.Linear(self.hidden_size, z_size)
        self.to_log_var = nn.Linear(self.hidden_size, z_size)
        self.z_to_hidden = nn.Linear(z_size, self.hidden_size)

    @torch.no_grad()
    def _get_flatten_size(self):
        x = self.encoder(self.example_input_array)
        return x.shape[-1]

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, log_var = self.to_mu(h), self.to_log_var(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        z, mu, log_var = self.bottleneck(h)
        z = self.z_to_hidden(z)
        z = z.view(z.size(0), self.hidden_size, 1, 1)  # Unflatten
        x_out = self.decoder(z)
        return x_out, mu, log_var
