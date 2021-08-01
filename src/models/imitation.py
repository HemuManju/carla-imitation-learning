import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Adam



def lossCriterion(inp, out):
    # value = nn.CrossEntropyLoss()
    # l1 = nn.MSELoss()
    # print('loss', len(inp), len(out))
    # print('loss2', inp[1].shape, out[1].shape)
    l1 = nn.functional.mse_loss(inp[0], out[0])
    l2 = nn.functional.cross_entropy(inp[1], out[1])
    # l3 = nn.functional.l1_loss(inp[2], out[2])
    loss = l1 + l2# + l3
    return loss



class Imitation(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Imitation, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader
        self.criterion = lossCriterion
        self.criterion = lossCriterion

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        target = [x, y]
        loss = self.criterion(output, target)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        # print('dims', x.shape, output[1].shape, y.shape)
        # print('dimsout', output[0].shape, output[1].shape, output[2].shape)
        target = [x,y]
        # print('valdim', target[0].shape, target[1].shape)
        loss = self.criterion(output, target)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['train_dataloader']

    def val_dataloader(self):
        return self.data_loader['val_dataloader']

    def test_dataloader(self):
        return self.data_loader['test_dataloader']

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    def scale_image(self, img):
        out = (img + 1) / 2
        return out
