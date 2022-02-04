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


def lossRNNCriterion(obj, inp, out, validation=False):
    '''The auxiliary loss function
    '''
    losses={} 
    # model output: out_seg, tl_state_output, dist_to_tl_output, dist_to_frontcar, act, rnn_input, rnn_out

    ### latent losses
    # losses['latent'] = nn.functional.smooth_l1_loss(inp[6], inp[5][:,1:,...])  # latent loss

    ### classification
    # losses['image_r'] = 1 - ms_ssim(inp[0], out[0][0], data_range=1, size_average=True)   # image reconstruction
    # losses['semantic'] = nn.functional.cross_entropy(inp[0], out[0][2])                   # pixel-wise semantic segmentation

    ### Class balanced
    losses['act'] = nn.functional.cross_entropy(inp[4], out[1][:,-1,1], obj.cb_weight['act'])   # Autopilot Action
    # losses['tr_status'] = nn.functional.cross_entropy(inp[1], out[1][:,0], obj.cb_weight['tr_status'])  # trafficlight status detection
    # losses['tr_dist'] = nn.functional.cross_entropy(inp[2], out[1][:,2], obj.cb_weight['tr_dist'])      # dist to traffic light
    # losses['car_dist'] = nn.functional.cross_entropy(inp[3], out[1][:,3], obj.cb_weight['car_dist'])    # dist to front car

    loss = 0
    for _, (k, val) in enumerate(losses.items()):
        # print('{}: {:.2f}'.format(k, val.item()), end='\t')
        if not validation:
            obj.log(k+'_loss', val.item(), on_step=False, on_epoch=True)
        else:
            obj.log(k+'_loss_val', val.item(), on_step=False, on_epoch=True)
        loss += val * obj.loss_factor[k]
    # print()

    return loss


def lossCriterion(obj, inp, out, validation=False):
    '''The auxiliary loss function
    '''
    losses={} 
    # model output: out_seg, tl_state_output, dist_to_tl_output, dist_to_frontcar

    ### classification
    # losses['image_r'] = 1 - ms_ssim(inp[0], out[0][0], data_range=1, size_average=True)   # image reconstruction
    # losses['semantic'] = nn.functional.cross_entropy(inp[0], out[0][2])                   # pixel-wise semantic segmentation

    ### Class balanced
    losses['act'] = nn.functional.cross_entropy(inp[4], out[1][:,1], obj.cb_weight['act'])   # Autopilot Action
    # losses['tr_status'] = nn.functional.cross_entropy(inp[1], out[1][:,0], obj.cb_weight['tr_status'])  # trafficlight status detection
    # losses['tr_dist'] = nn.functional.cross_entropy(inp[2], out[1][:,2], obj.cb_weight['tr_dist'])      # dist to traffic light
    # losses['car_dist'] = nn.functional.cross_entropy(inp[3], out[1][:,3], obj.cb_weight['car_dist'])    # dist to front car

    loss = 0
    for _, (k, val) in enumerate(losses.items()):
        # print('{}: {:.2f}'.format(k, val.item()), end='\t')
        if not validation:
            obj.log(k+'_loss', val.item(), on_step=False, on_epoch=True)
        else:
            obj.log(k+'_loss_val', val.item(), on_step=False, on_epoch=True)
        loss += val * obj.loss_factor[k]
    # print()

    return loss



class Imitation(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Imitation, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader
        self.criterion = lossRNNCriterion #lossCriterion
        self.before_train()

        self.val_loss_in_valStep = True

    def forward(self, x):
        output = self.net.forward(x)
        return output
    
    def before_train(self):

        ### logging
        info = ('\n' + str(datetime.now()) +': Just Image Reconstruction + No Data Augmentation  + reg=0 + NO action' #+
         #' + 256 latent CB_LOSS for all except semseg + Factor 3 for semseg\n'
        )
        print(info)
        with open(self.h_params.log_dir + 'info.txt', 'a') as f:
            f.write(info)

        ### class-balanced loss
        distributions = {
            'tr_status':{0: 118975, 1: 35064, 2: 356261},
            'tr_dist': {0: 40710, 1: 54691, 2: 28645, 3: 386254},
            'car_dist':{0: 91403, 1: 17656, 2: 11073, 3: 390168},
            'act': {0: 5859, 1: 27987, 2: 230868, 3: 23617, 4: 7574, 5:0, 6: 1569, 7: 683, 8: 316, 9:0, 10: 13136,
             11: 20619, 12: 101168, 13: 16284, 14: 5649, 15: 2284, 16: 18450, 17: 13646, 18: 19437, 19: 1154}
        }
        self.cb_weight = {}
        for _, (key, val) in enumerate(distributions.items()):
            occurences = np.array(list(val.values()))
            occurences[np.where(occurences==0)] = np.max(occurences)
            class_weight = 1.0/occurences
            self.cb_weight[key] = torch.from_numpy(class_weight).type(torch.float32).to(torch.device('cuda:0'))
        self.loss_factor={'semantic':3, 'act':1, 'image_r':1, 'tr_status':1 ,'tr_dist':1 , 'car_dist': 1, 'latent': 3}

    def on_train_start(self) -> None:
        self.val_loss_in_valStep = True

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.h_params['use_hlcmd']:
            _, sorted_ind = torch.sort(x[1][:,0], stable=True)
            x[0] = x[0][sorted_ind]
            x[1] = x[1][sorted_ind]
            x[2] = x[2][sorted_ind]
            y = y[sorted_ind]

        # Predict and calculate loss
        output = self.forward(x)
        target = [x, y]
        loss = self.criterion(self, output, target)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.h_params['use_hlcmd']:
            _, sorted_ind = torch.sort(x[1][:,0], stable=True)
            x[0] = x[0][sorted_ind]
            x[1] = x[1][sorted_ind]
            x[2] = x[2][sorted_ind]

            y = y[sorted_ind]

        # Predict and calculate loss
        output = self.forward(x)
        target = [x,y]

        if self.val_loss_in_valStep:
            loss = self.criterion(self, output, target, validation=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        else:  # To Log loss before training
            loss = self.criterion(self, output, target, validation=False)
            self.log('train_loss', loss, on_step=False, on_epoch=True)
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


    def calcAccuracy(self, dataset_type='val'):
        ''' calculate the per-class accuaracy and plot confusion matrix
        NOTE: must have <255 classes or else remove astype(np.uint8) used to reduce size
        '''
       
        ### select keys in the auxiliary model
        # keys = ['dist_car', 'traffic', 'dist_tr']
        keys = ['act']
        # keys = ['semseg']
        
        self.net.eval()
        with torch.no_grad():
            net = self.net.to(torch.device('cuda:0'))
            dataloader = self.data_loader[dataset_type+'_dataloader']
            # intialization
            pred_bt = dict.fromkeys(keys)
            labels_bt = dict.fromkeys(keys)
            pred_total = {key:[] for key in keys}
            labels_total = {key:[] for key in keys}

            for i, batch in enumerate(dataloader):
                x, y = batch
                x[0] = x[0].to(torch.device('cuda:0'))
                x[1] = x[1].to(torch.device('cuda:0'))

                # Pass through the network
                output = net(x)
                target = [x,y]      # to match the loss criterion format

                ### calculate accuracy  (fix the indexes here-look at loss function)
                if 'act' in keys:
                    pred_bt['act'] = torch.argmax(output[4],dim=1).cpu()
                    labels_bt['act'] = target[1][:,1].cpu()
                if 'dist_car' in keys:
                    pred_bt['dist_car'] = torch.argmax(output[3],dim=1).cpu()
                    labels_bt['dist_car'] = target[1][:,3].cpu()
                if 'traffic' in keys:
                    pred_bt['traffic'] = torch.argmax(output[1],dim=1).cpu()
                    labels_bt['traffic'] = target[1][:,0].cpu()
                if 'dist_tr' in keys:
                    pred_bt['dist_tr'] = torch.argmax(output[2],dim=1).cpu()
                    labels_bt['dist_tr'] = target[1][:,2].cpu()
                if 'semseg' in keys:
                    pred_bt['semseg'] = torch.argmax(output[0],dim=1).cpu()
                    labels_bt['semseg'] = target[0][2].cpu()
                
                
                print('\nbatch {}/{} acc - '.format(i,len(dataloader)), end='')
                for key in keys:
                    pred_total[key].append(pred_bt[key].view(-1).numpy().astype(np.uint8))      # cast to uint8 to reduce size
                    labels_total[key].append(labels_bt[key].view(-1).numpy().astype(np.uint8))
                    batchacc = 100*accuracy_score(labels_bt[key].view(-1), pred_bt[key].view(-1))
                    print('{}: {:.1f}'.format(key, batchacc), end='\t')


            for key in keys:
                print('\n*******************\nAux Task:',key)
                labels_total_flat = np.concatenate(labels_total[key]).ravel()
                pred_total_flat = np.concatenate(pred_total[key]).ravel()

                conf_mat = confusion_matrix(labels_total_flat, pred_total_flat)
                print(conf_mat)
                # Per-class accuracy
                perclass_acc = np.around(100*conf_mat.diagonal()/conf_mat.sum(1), decimals=1)
                acc = 100*accuracy_score(labels_total_flat, pred_total_flat)
                print('Per-class accuracy:', perclass_acc)
                print('Accuracy:{:.2f}'.format(acc))
                
                disp = ConfusionMatrixDisplay(conf_mat)
                disp.plot(cmap=plt.cm.Blues)
                plt.title('Task {}: acc={:.2f}% - per-class acc=\n{}'.format(key, acc, np.array2string(perclass_acc)))
                plt.savefig('data/confusionMat_'+key+'.png')
                plt.show()


    def sampleOutput(self, dataset_type='val'):
        dataloader = self.data_loader[dataset_type + '_dataloader']
        plotSemseg = dataloader.dataset.plotSemseg      # for plotting the semseg
        self.net.eval()
        with torch.no_grad():
            net = self.net.to(torch.device('cuda:0'))

            array2, array3 = None, None
            f, axarr = plt.subplots(2,2)
            for i, batch in enumerate(dataloader):
                x, y = batch
                x[0] = x[0].to(torch.device('cuda:0'))
                x[1] = x[1].to(torch.device('cuda:0'))
                output = net(x)
                b_ind = random.randint(0, self.h_params['BATCH_SIZE']-1)        # index in current batch 

                ### plot image (multiple images mode)
                # ind = 0
                # array1 = x[0][b_ind][ind].cpu().detach().numpy()
                # array2 = output[0][b_ind][ind].cpu().detach().numpy()

                ### plot single original input image
                array1 = x[0][b_ind][:].cpu().detach().numpy()
                array1 = 255 * np.moveaxis(array1, 0, 2)

                #################################
                ### reconstructed image
                array2 = output[0][b_ind][:].cpu().detach().numpy()
                array2 = 255 * np.moveaxis(array2, 0, 2)
                
                #################################
                # ### semseg model
                # array2 = torch.argmax(output[0][b_ind][:],dim=0).cpu().detach().numpy()
                # array2 = plotSemseg(array2)

                # ### semseg ground truth
                # array3 = x[2][b_ind][:].cpu().detach().numpy()
                # array3 = plotSemseg(array3)
   

                ###################################
                ###################################
                ### Plot the image
                axarr[0,0].imshow(np.uint8(array1),)
                axarr[0,0].title.set_text('original image')

                if array3 is not None:                
                    axarr[1,0].imshow(np.uint8(array3),)
                    axarr[1,0].title.set_text('ground-truth')
                
                if array2 is not None:  
                    axarr[1,1].imshow(np.uint8(array2),)
                    axarr[1,1].title.set_text('semseg output')


                # axarr[0].imshow(np.uint8(array1*255), cmap='gray',)
                # axarr[1].imshow(np.uint8(array2*255), cmap='gray',)
                plt.show(block=False)
                input("Press Enter to continue...")
    