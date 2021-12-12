from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from datetime import datetime
from pytorch_msssim import ms_ssim

from torch.optim import Adam
import random

from matplotlib import pyplot as plt



def lossCriterion(obj, inp, out):
    l_act, l_sem, l_tr, l_tr_dist, l_car_dist = 0, 0, 0, 0, 0 

    # model output: out_seg, tl_state_output, dist_to_tl_output, dist_to_frontcar
    ### calculate loss
    # regression
    # l1 = nn.functional.mse_loss(inp[0], out[0][0])                              # image reconstruction MSE
    # l1 = 1 - ms_ssim(inp[0], out[0][0], data_range=1, size_average=True)        # image reconstruction
    # l1 = 1 - ms_ssim(inp[0], out[0][2], data_range=1, size_average=True)        # semantic segmentation
    
    # classification
    # l_act = nn.functional.cross_entropy(inp[2], out[1][:,1])                      # Autopilot Action
    l_sem = nn.functional.cross_entropy(inp[0], out[0][2])                        # pixel-wise semantic segmentation
    l_tr = nn.functional.cross_entropy(inp[1], out[1][:,0])                       # trafficlight status detection
    l_tr_dist = nn.functional.cross_entropy(inp[2], out[1][:,2])                  # dist to traffic light
    l_car_dist = nn.functional.cross_entropy(inp[3], out[1][:,3])                 # dist to front car
   
    ### plot loss
    # print('loss:', l_sem.item(), l_tr.item(), l_tr_dist.item(), l_car_dist.item())
    # obj.log('autopilot_action_loss', l_act.item(), on_step=False, on_epoch=True)
    obj.log('image_loss', l_sem.item(), on_step=False, on_epoch=True)
    obj.log('traffic_loss', l_tr.item(), on_step=False, on_epoch=True)
    obj.log('traffic_dist_loss', l_tr_dist.item(), on_step=False, on_epoch=True)
    obj.log('frontcar_dist_loss', l_car_dist.item(), on_step=False, on_epoch=True)


    ### weighted summation
    loss = l_sem + l_tr + l_tr_dist + l_car_dist
    return loss



class Imitation(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Imitation, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader
        self.criterion = lossCriterion
        self.before_train()

    def forward(self, x):
        output = self.net.forward(x)
        return output
    
    def before_train(self):
        cur_time = str(datetime.now())
        info = cur_time +': just all aux encoder + single img in single out + PRETRAINED RESNET+ NO action\n'
        print(info)
        with open(self.h_params.log_dir + 'info.txt', 'a') as f:
            f.write(info)
        class_weight = 1.0/np.array([45150, 21977, 17134, 181751])
        self.cb_weight = torch.from_numpy(class_weight).type(torch.float32).to(torch.device('cuda:0'))

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
            net = self.net.to(torch.device('cuda:0'))
            dataloader = self.data_loader['train_dataloader']
            # keys in the auxiliary model
            keys = ['act']
            # intialization
            pred_bt = dict.fromkeys(keys)
            labels_bt = dict.fromkeys(keys)
            pred_total = dict.fromkeys(keys, torch.zeros(0,dtype=torch.long, device='cpu'))
            labels_total = dict.fromkeys(keys, torch.zeros(0,dtype=torch.long, device='cpu'))

            for i, batch in enumerate(dataloader):
                x, y = batch
                x[0] = x[0].to(torch.device('cuda:0'))

                # Pass through the network
                output = net(x)
                x = 0  # to match lossCriterion format
                target = [x,y]

                ### calculate accuracy
                if 'act' in keys:
                    pred_bt['act'] = torch.argmax(output[2],dim=1).cpu()
                    labels_bt['act'] = target[1][:,1].cpu()
                if 'dist' in keys:
                    pred_bt['dist'] = torch.argmax(output[3],dim=1).cpu()
                    labels_bt['dist'] = target[1][:,2].cpu()


                print('\nbatch {}/{} acc - '.format(i,len(dataloader)), end='')
                for key in keys:
                    pred_total[key] = torch.cat((pred_total[key], pred_bt[key].view(-1)))
                    labels_total[key] = torch.cat((labels_total[key], labels_bt[key].view(-1)))
                    batchacc = 100*accuracy_score(labels_bt[key].view(-1), pred_bt[key].view(-1))
                    print('{}: {:.1f}'.format(key, batchacc), end='\t')


            for key in keys:
                print('\n*******************\nAux Task:',key)
                conf_mat = confusion_matrix(labels_total[key].numpy(), pred_total[key].numpy())
                print(conf_mat)
                # Per-class accuracy
                perclass_acc = np.around(100*conf_mat.diagonal()/conf_mat.sum(1), decimals=1)
                acc = 100*accuracy_score(labels_total[key].numpy(), pred_total[key].numpy())
                print('Per-class accuracy:', perclass_acc)
                print('Accuracy:{:.2f}'.format(acc))
                
                disp = ConfusionMatrixDisplay(conf_mat)
                disp.plot(cmap=plt.cm.Blues)
                plt.title('Task {}: acc={:.2f}% - per-class acc=\n{}'.format(key, acc, np.array2string(perclass_acc)))
                plt.savefig('data/confusionMat_'+key+'.png')
                plt.show()


    def sampleOutput(self):
        self.net.eval()
        with torch.no_grad():
            net = self.net.to(torch.device('cuda:0'))
            dataloader = self.data_loader['val_dataloader']
            # object_methods = [method_name for method_name in dir(dataloader)
            #       if callable(getattr(dataloader, method_name))]
            # print(object_methods)


            # data = dataloader.y
            # print(data.shape)

            # plt.hist(data[:,2], density=True, bins=30)  # density=False would make counts
            # plt.ylabel('Probability')
            # plt.xlabel('Data');

            f, axarr = plt.subplots(1,2)
            for i, batch in enumerate(dataloader):
                x, y = batch
                x[0] = x[0].to(torch.device('cuda:0'))
                output = net(x)
                ind = 0
                b_ind = random.randint(0, self.h_params['BATCH_SIZE']-1)
                array1 = x[0][b_ind][ind].cpu().detach().numpy()
                array2 = output[0][b_ind][ind].cpu().detach().numpy()

                axarr[0].imshow(np.uint8(array1*255), cmap='gray',)
                axarr[1].imshow(np.uint8(array2*255), cmap='gray',)
                plt.show(block=False)
                input("Press Enter to continue...")
            
