from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_msssim import ms_ssim

from torch.optim import Adam
import random

from matplotlib import pyplot as plt



def lossCriterion(obj, inp, out):
    ### calculate loss
    # l1 = nn.functional.mse_loss(inp[0], out[0][0])                        # image reconstruction MSE
    # l1 = 1 - ms_ssim(inp[0], out[0][0], data_range=1, size_average=True)  # image reconstruction
    l1 = 1 - ms_ssim(inp[0], out[0][2], data_range=1, size_average=True)    # semantic segmentation
    # l2 = nn.functional.cross_entropy(inp[1], out[1][:,0])                 # trafficlight status detection
    l3 = nn.functional.cross_entropy(inp[2], out[1][:,1])                 # Autopilot Action
   
    ### plot loss
    # print('loss:',l1.item(), l2.item(), l3.item())
    obj.log('image_recons_loss', l1.item(), on_step=False, on_epoch=True)
    obj.log('traffic_loss', l2.item(), on_step=False, on_epoch=True)
    obj.log('autopilot_action_loss', l3.item(), on_step=False, on_epoch=True)

    ### weighted summation
    # loss = 2*l1 + l2 + l3
    # loss = l2 + 3*l3
    loss = 2*l1 + l3
    # loss = l3
    # loss = l1
    return loss



class Imitation(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Imitation, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader
        self.criterion = lossCriterion
        # self.criterion = nn.CrossEntropyLoss()
        self.log('info', 'this is the info',on_step=False, on_epoch=True)

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        target = [x, y]
        loss = self.criterion(self, output, target)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output = self.forward(x)
        target = [x,y]
        loss = self.criterion(self, output, target)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['train_dataloader']

    def val_dataloader(self):
        return self.data_loader['val_dataloader']

    def test_dataloader(self):
        return self.data_loader['test_dataloader']

    def configure_optimizers(self):
        return Adam(self.parameters(),
                    lr = self.h_params['LEARNING_RATE'],
                    weight_decay = self.h_params['WEIGHT_DECAY']
                    )

    def scale_image(self, img):
        out = (img + 1) / 2
        return out


    def calcAccuracy(self):
        self.net.eval()
        with torch.no_grad(): 
            dataloader = self.data_loader['train_dataloader']
            # train_features, train_labels = next(iter(train_dataloader))
            corrects_total = {'tr':0, 'act':0}

            labels = []
            predicted = []
            for i, batch in enumerate(dataloader):
                x, y = batch

                # Pass through the network
                output = self.net(x)
                target = [x,y]

                ### calculate accuracy
                # corrects_1 = (torch.argmax(output[1],dim=1) == target[1][:,0]).sum()  # trafficlight status detection
                corrects_2 = (torch.argmax(output[2],dim=1) == target[1][:,1]).sum()    # autopilot action detection
                corrects_1 = 0                                                          # if no aux task


                # predicted.append(torch.argmax(output[1],dim=1).cpu().detach().numpy())
                # labels.append(y.cpu().detach().numpy())
                # print(torch.argmax(output[1],dim=1).cpu().detach().numpy())
                # print(target[1][:,0].cpu().detach().numpy())
                
                corrects_total['tr'] += corrects_1
                corrects_total['act'] += corrects_2
                print('batch {}/{} - batch acc: {:.3f}\t{:.3f}'.format(i,len(dataloader),
                float(corrects_1/target[1][:,1].shape[0]), float(corrects_2/target[1][:,1].shape[0])))

            # predicted = np.concatenate(predicted, axis=0 )
            # labels =  np.concatenate(labels, axis=0 )
            # np.save('predWlabels.npy',{'predicted':predicted, 'labels':labels}, allow_pickle=True)
            
            print('accuracy total (traffic light): {}/{}'.format(corrects_total['tr'], 64*dataloader.__len__()))
            print('accuracy total (autopilot action): {}/{}'.format(corrects_total['act'], 64*dataloader.__len__()))
            print('accuracy total: {}/{}'.format(corrects_total['tr'], self.h_params['BATCH_SIZE']*dataloader.__len__()))

    def sampleOutput(self):
        self.net.eval()
        with torch.no_grad():
            dataloader = self.data_loader['val_dataloader']
            # object_methods = [method_name for method_name in dir(dataloader)
            #       if callable(getattr(dataloader, method_name))]
            # print(object_methods)


            # data = dataloader.giveData()
            # print(data.shape)

            # plt.hist(data[:,0], density=True, bins=30)  # density=False would make counts
            # plt.ylabel('Probability')
            # plt.xlabel('Data');

            f, axarr = plt.subplots(1,2)
            for i, batch in enumerate(dataloader):
                x, y = batch
                output = self.net(x)
                ind = 0
                b_ind = random.randint(0, self.h_params['BATCH_SIZE']-1)
                array1 = x[0][b_ind][ind].cpu().detach().numpy()
                array2 = output[0][b_ind][ind].cpu().detach().numpy()

                axarr[0].imshow(np.uint8(array1*255), cmap='gray',)
                axarr[1].imshow(np.uint8(array2*255), cmap='gray',)
                plt.show(block=False)
                input("Press Enter to continue...")
            
