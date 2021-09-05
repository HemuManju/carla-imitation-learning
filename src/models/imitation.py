import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Adam, lr_scheduler

from matplotlib import pyplot as plt


def lossCriterion(obj, inp, out):
    # l1 = nn.functional.mse_loss(inp[0], out[0][0])           # image reconstruction
    # l2 = nn.functional.cross_entropy(inp[1], out[1][:,0])    # trafficlight status detection
    l3 = nn.functional.cross_entropy(inp[2], out[1][:, 1])  # Autopilot Action

    # print('loss:',l1.item(), l2.item(), l3.item())
    # obj.log('image_recons_loss', l1.item(), on_step=False, on_epoch=True)
    # obj.log('traffic_loss', l2.item(), on_step=False, on_epoch=True)
    # obj.log('autopilot_action_loss', l3.item(), on_step=False, on_epoch=True)

    # loss = l1 + 0.25*l2 + l3
    # loss = l1 + l3
    loss = l3
    return loss


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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        self.log('val_loss', loss)  # needed for checkpointing
        return loss

    def training_epoch_end(self, outputs) -> None:
        # Update scheduler
        sch = self.lr_schedulers()
        sch.step()

        # Calculate mean loss
        temp = torch.stack([item['loss'] for item in outputs])
        loss = torch.mean(temp)
        self.logger.experiment.add_scalars("losses", {"train_loss": loss},
                                           global_step=self.current_epoch)

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.mean(torch.stack(outputs))
        self.logger.experiment.add_scalars("losses", {"val_loss": loss},
                                           global_step=self.current_epoch)

    def train_dataloader(self):
        return self.data_loader['train_dataloader']

    def val_dataloader(self):
        return self.data_loader['val_dataloader']

    def test_dataloader(self):
        return self.data_loader['test_dataloader']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[20, 30],
                                             gamma=0.1)
        return [optimizer], [scheduler]

    def scale_image(self, img):
        out = (img + 1) / 2
        return out


class ImitationAux(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(ImitationAux, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader
        self.criterion = lossCriterion
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        target = [x, y]
        loss = self.criterion(self, output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        target = [x, y]
        loss = self.criterion(self, output, target)
        return loss

    def training_epoch_end(self, outputs) -> None:
        # Update scheduler
        sch = self.lr_schedulers()
        sch.step()

        # Calculate mean loss
        temp = torch.stack([item['loss'] for item in outputs])
        loss = torch.mean(temp)
        self.logger.experiment.add_scalars("losses", {"train_loss": loss},
                                           global_step=self.current_epoch)

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.mean(torch.stack(outputs))
        self.logger.experiment.add_scalars("losses", {"val_loss": loss},
                                           global_step=self.current_epoch)

    def train_dataloader(self):
        return self.data_loader['train_dataloader']

    def val_dataloader(self):
        return self.data_loader['val_dataloader']

    def test_dataloader(self):
        return self.data_loader['test_dataloader']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[20, 30],
                                             gamma=0.1)
        return [optimizer], [scheduler]

    def scale_image(self, img):
        out = (img + 1) / 2
        return out

    def calc_accuracy(self):
        self.net.eval()
        with torch.no_grad():
            dataloader = self.data_loader['val_dataloader']
            # train_features, train_labels = next(iter(train_dataloader))
            corrects_total = 0

            labels = []
            predicted = []
            for i, batch in enumerate(dataloader):
                x, y = batch

                # Predict and calculate loss
                output = self.net(x)
                # ## Auxiliary architechture
                target = [x, y]
                sortedout = torch.argmax(output[1], dim=1)
                corrects = (sortedout == target[1]).sum()

                # Regular architechture
                # sortedout = torch.argmax(output, dim=1)
                # corrects = (sortedout == y).sum()
                # print(sortedout, y)

                predicted.append(sortedout.cpu().detach().numpy())
                labels.append(y.cpu().detach().numpy())
                corrects_total += corrects
                print('batch {}/{} - batch acc: {}'.format(
                    i, len(dataloader), float(corrects / sortedout.shape[0])))

            predicted = np.concatenate(predicted, axis=0)
            labels = np.concatenate(labels, axis=0)
            np.save('predWlabels.npy', {
                'predicted': predicted,
                'labels': labels
            },
                    allow_pickle=True)

            print('accuracy total: {}/{}'.format(corrects_total,
                                                 64 * dataloader.__len__()))
            print('accuracy total: {}/{}'.format(
                corrects_total,
                self.h_params['BATCH_SIZE'] * dataloader.__len__()))

    def sample_output(self):
        self.net.eval()
        with torch.no_grad():
            dataloader = self.data_loader['val_dataloader']
            object_methods = [
                method_name for method_name in dir(dataloader)
                if callable(getattr(dataloader, method_name))
            ]
            print(object_methods)

            data = dataloader.giveData()
            print(data.shape)

            plt.hist(data[:, 0], density=True,
                     bins=30)  # density=False would make counts
            plt.ylabel('Probability')
            plt.xlabel('Data')

            # f, axarr = plt.subplots(1,2)
            # for i, batch in enumerate(dataloader):
            #     x, y = batch
            #     output = self.net(x)
            #     ind = 0
            #     b_ind = random.randint(0, self.h_params['BATCH_SIZE']-1)
            #     array1 = x[b_ind][ind].cpu().detach().numpy()
            #     array2 = output[0][b_ind][ind].cpu().detach().numpy()

            #     axarr[0].imshow(np.uint8(array1*255), cmap='gray',)
            #     axarr[1].imshow(np.uint8(array2*255), cmap='gray',)
            #     plt.show(block=False)
            #     input("Press Enter to continue...")
