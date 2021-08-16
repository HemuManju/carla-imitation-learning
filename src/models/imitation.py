from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Adam
import random

from matplotlib import pyplot as plt



def lossCriterion(inp, out):
    # value = nn.CrossEntropyLoss()
    # l1 = nn.MSELoss()
    # print('loss', len(inp), len(out))
    # print('loss2', inp[1].shape, out[1].shape)
    l1 = nn.functional.mse_loss(inp[0], out[0])
    l2 = nn.functional.cross_entropy(inp[1], out[1])
    print('loss:',l1.item(), l2.item())
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
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        target = [x, y]
        loss = self.criterion(output, target)
        # loss = self.criterion(output, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        # print('dims', x.shape, output[1].shape, y.shape)
        # print('dimsout', output[0].shape, output[1].shape, output[2].shape)
        target = [x,y]
        loss = self.criterion(output, target)
        # loss = self.criterion(output, y)

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


    def calcAccuracy(self):
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
                ### Auxiliary architechture
                # target = [x,y]
                # sortedout = torch.argmax(output[1], dim=1)
                # corrects = (sortedout == target[1]).sum()

                ### Regular architechture 
                sortedout = torch.argmax(output, dim=1)
                corrects = (sortedout == y).sum()
                # print(sortedout, y)

                predicted.append(sortedout.cpu().detach().numpy())
                labels.append(y.cpu().detach().numpy())
                corrects_total += corrects
                print('batch {}/{} - batch acc: {}'.format(i,len(dataloader),float(corrects/sortedout.shape[0])))

            predicted = np.concatenate( predicted, axis=0 )
            labels =  np.concatenate( labels, axis=0 )
            np.save('predWlabels.npy',{'predicted':predicted, 'labels':labels}, allow_pickle=True)
            
            print('accuracy total: {}/{}'.format(corrects_total, 64*dataloader.__len__()))
            print('accuracy total: {}/{}'.format(corrects_total, self.h_params['BATCH_SIZE']*dataloader.__len__()))

    def sampleOutput(self):
        self.net.eval()
        with torch.no_grad():
            dataloader = self.data_loader['val_dataloader']
            object_methods = [method_name for method_name in dir(dataloader)
                  if callable(getattr(dataloader, method_name))]
            print(object_methods)


            data = dataloader.giveData()
            print(data.shape)

            plt.hist(data[:,0], density=True, bins=30)  # density=False would make counts
            plt.ylabel('Probability')
            plt.xlabel('Data');

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
            
