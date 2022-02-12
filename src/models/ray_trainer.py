import os

import torch
import torch.nn as nn

from ray import tune
from ray.tune.integration.torch import (
    DistributedTrainableCreator,
    distributed_checkpoint_dir,
)

from .utils import train, test


class RayTrainer:
    def __init__(self, hparams, net, data_loader, optimizer, criterion):
        self.hparams = hparams
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.net = self._setup_net(net)

        # Check the data loaders
        self._check_data_loader()

    def _check_data_loader(self):
        default_keys = ['train_data_loader', 'val_data_loader', 'test_data_loader']
        keys = list(self.data_loader.keys())
        assert keys in default_keys, "Data loader is missing train/val/test dataset"

    def _setup_net(self, net):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        return net.to(device)

    def fit(self, num_workers=0, num_gpus_per_worker=0, workers_per_node=None):
        if torch.cuda.device_count() > 1:
            trainable_cls = DistributedTrainableCreator(
                self._train,
                num_workers=num_workers,
                num_gpus_per_worker=num_gpus_per_worker,
                num_workers_per_host=workers_per_node,
            )
        else:
            trainable_cls = self._train

        tune.run(
            trainable_cls,
            num_samples=4,
            stop={"training_iteration": 10},
            metric="mean_accuracy",
            mode="max",
        )

    def _train(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        for epoch in range(self.hparams['epochs']):
            train(
                self.net,
                self.data_loader['train_data_loader'],
                self.optimizer,
                self.criterion,
                device,
            )

            test(self.net, self.data_loader['val_data_loader'], self.criterion, device)

            if epoch % self.hparams['log_freq'] == 0:
                if torch.cuda.device_count() > 1:
                    with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save(
                            (self.model.state_dict(), self.optimizer.state_dict()), path
                        )
                else:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(
                        (self.model.state_dict(), self.optimizer.state_dict()), path
                    )
