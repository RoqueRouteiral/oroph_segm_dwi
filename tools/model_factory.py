import os
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import nibabel as nib
import scipy

from tools.misc import thumbnail

# Segmentation models
from models.unet3d import Unet3d
from models.YNet import YNet
from models.YNet_mp import YNet_mp
from models.XNet import XNet
from models.XNet_mp import XNet_mp
from models.XNet_linear import XNet_linear
from models.latef import Xlate
from models.latef_mp import Xlate_mp
from models.unet_small import Small_u
from models.unet_det import Unet_det


from tools.metrics import dice, DiceLoss
from tools.metrics_dm import dice_dm, hd_dm, msd, compute_surface_distances_dm

from tools.plots import plot_history, plot_history_offline


import torch.optim as optim
#from torchsummary import summary

# Build the model
class Model():
    
    def __init__(self, cf):
        self.cf = cf
        self.device = torch.device("cuda:"+str(self.cf['gpu']) if torch.cuda.is_available() else "cpu")
        self.savepath = os.path.join(self.cf['experiments'],self.cf['exp_name'])
        if self.cf['model_name'] == 'unet3d':
            self.model = Unet3d(self.cf['n_channels'])   
        elif self.cf['model_name'] == 'ynet':
            self.model = YNet(1)   
        elif self.cf['model_name'] == 'ynet_mp':
            self.model = YNet_mp(1)   
        elif self.cf['model_name'] == 'xnet':
            self.model = XNet(1)   
        elif self.cf['model_name'] == 'xnet_mp':
            self.model = XNet_mp(1)   
        elif self.cf['model_name'] == 'xnet_linear':
            self.model = XNet_linear(1)   
        elif self.cf['model_name'] == 'xlate':
            self.model = Xlate(1)    
        elif self.cf['model_name'] == 'xlate_mp':
            self.model = Xlate_mp(1)   
        elif self.cf['model_name'] == 'small':
            self.model = Small_u(self.cf['n_channels'])  
        elif self.cf['model_name'] == 'det_u':
            self.model = Unet_det(self.cf['n_channels'])  
        else:
            raise ValueError('Unknown model')
        # Show model structure
#        if self.cf.cuda:
#            self.model.cuda()
        self.counter=0
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, [self.cf['gpu']])
        self.model.to(self.device)

        # Output the model
        print ('   Model: ' + self.cf['model_name'])
         

    
    def train(self, train_gen, val_gen):
        if self.cf['finetune']:
            print('Loading weights for finetuning')
            self.model.load_state_dict(torch.load(self.cf['weights_test_finetune']))        
            #Defining Loss, optimizer and Lr-scheduler
        if self.cf['loss']=='dice_loss':
            self.criterion =DiceLoss()
            self.criterion2 = nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cf['learning_rate'])
        best_id_epoch = 0
        if self.cf['lr_scheduler']:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.001)
            #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.5)
        #Starting the training
        #self.alphas=np.concatenate((np.zeros(round(self.cf.epochs/4)),np.linspace(0,1,round(self.cf.epochs/2)),np.ones(round(self.cf.epochs/4))))
        train_losses=np.zeros(self.cf['epochs'])
        train_dices=np.zeros(self.cf['epochs'])
        val_losses=np.zeros(self.cf['epochs'])
        val_dices=np.zeros(self.cf['epochs'])
        best_dice = 0

        for epoch in range(self.cf['epochs']):
            start_time = time.time()
            train_losses[epoch], train_dices[epoch]=self._one_epoch(train_gen,'train')
            val_losses[epoch], val_dices[epoch]=self._one_epoch(val_gen,'eval')                    
            print('[%d/%d] loss: %.3f dice %.3f val_loss: %.3f val_dice: %.3f time: %.3f min' % (epoch + 1, self.cf['epochs'], train_losses[epoch], train_dices[epoch], val_losses[epoch], val_dices[epoch], (time.time() - start_time)/60))
#            print('Using Enhancing of {}'.format(self.list_of_factors[-1]))
            if self.cf['lr_scheduler']:    
                self.scheduler.step(train_losses[epoch])
            if not epoch%self.cf['snapshots']:
                torch.save(self.model.state_dict(), (self.savepath+'/snapshot_epoch{}.ckpt'.format(epoch)))
                if epoch>0:
                    plot_history(train_losses[0:epoch], train_dices[0:epoch], val_losses[0:epoch], val_dices[0:epoch], self.savepath+'/hist_epoch{}.png'.format(epoch))
            if val_dices[epoch]>best_dice:
                best_id_epoch = epoch
                best_dice = val_dices[epoch]
                np.save(self.savepath+'/best_epoch.npy',best_id_epoch)
                np.save(self.savepath+'/best_dice.npy',best_dice)
                torch.save(self.model.state_dict(), self.savepath+'/best_model.ckpt')

        torch.save(self.model.state_dict(), self.savepath+'/trained_model.ckpt')
        #saving epochs
        np.save(self.savepath+'/train_losses.npy',train_losses)
        np.save(self.savepath+'/train_dices.npy',train_dices)
        np.save(self.savepath+'/val_losses.npy',val_losses)
        np.save(self.savepath+'/val_dices.npy',val_dices)
        plot_history(train_losses, train_dices, val_losses, val_dices, self.savepath+'/hist.png')
        print('Finished training')


    
    def repeat(self, train_gen, val_gen, extra_val_gen,path_to_models='D:/project_2/scripts/p2_segmentation/Experiments_dwi/box_dwi_bm/'):

        train_dices=np.zeros(50)
        val_dices=np.zeros(50)
        extra_val_dices=np.zeros(50)
        
        train_hds=np.zeros(50)
        val_hds=np.zeros(50)
        extra_val_hds=np.zeros(50)        
        this_save_path=self.savepath+'/offline_curve/'
        if not os.path.exists(this_save_path):
            os.makedirs(this_save_path)   
        idx=0
        for epoch in range(0,150,3):
            idx+=1
            self.model.load_state_dict(torch.load(path_to_models+'/snapshot_epoch{}.ckpt'.format(epoch)))
            train_dices[idx], train_hds[idx]=self._one_epoch_repeat(train_gen,'train')
            val_dices[idx], val_hds[idx]=self._one_epoch_repeat(val_gen,'eval')
            extra_val_dices[idx], extra_val_hds[idx]=self._one_epoch_repeat(extra_val_gen,'eval')                    
                    
            print('[%d/%d] train dice %.3f val dice: %.3f extra val dice: %.3f' % (epoch, 150, train_dices[idx], val_dices[idx],extra_val_dices[idx]))
            plot_history_offline(train_dices[0:idx], val_dices[0:idx], extra_val_dices[0:idx], train_hds[0:idx], val_hds[0:idx], extra_val_hds[0:idx], this_save_path+'/hist_offline.png')
            np.save(this_save_path+'train_dices.npy',train_dices[0:idx])
            np.save(this_save_path+'val_dices.npy',val_dices[0:idx])
            np.save(this_save_path+'extra_val_dices.npy',extra_val_dices[0:idx])
            
            np.save(this_save_path+'train_hds.npy',train_hds[0:idx])
            np.save(this_save_path+'val_hds.npy',val_hds[0:idx])
            np.save(this_save_path+'extra_val_hds.npy',extra_val_hds[0:idx])        
        
    def test(self, test_gen):
        self.model.load_state_dict(torch.load(self.cf['weights_test_file']))
        if self.cf['test_set']:
            thumb_path=self.savepath+'/inference_test/' 
        elif self.cf['val_set']:
            thumb_path=self.savepath+'/inference_validation/' 
        if self.cf['thumbnail']:
            if not os.path.exists(thumb_path):
                os.makedirs(thumb_path)
        print(len(test_gen))
        running_dice = np.zeros(len(test_gen))
        running_hd = np.zeros(len(test_gen))
        running_msd = np.zeros(len(test_gen))
        self.model.eval()
        if self.cf['resize']: 
            vox_spacing = [275/112*x for x in (0.79323106, 0.7936626, 0.78976499)]
        else:
            vox_spacing = [x for x in (0.79323106, 0.7936626, 0.78976499)]
        with torch.no_grad():        
            for i, (data, name) in enumerate(test_gen, 0):
                # get the inputs         
                if (self.cf['model_name'] == 'xnet'):
                    inputs,labels, labels_box = data
                else:
                    inputs, labels = data
                if ((self.cf['model_name'] == 'ynet') or (self.cf['model_name'] == 'xnet') or (self.cf['model_name'] == 'xlate')):
                    input_a =  inputs[:,0:3] #change
                    input_b = inputs[:,3].unsqueeze(0)
#                    print(input_a.shape, inputs.shape, input_b.shape)
                    outputs=(self.model(input_a.to(self.device),input_b.to(self.device)))
                    if ((self.cf['model_name'] == 'ynet') or (self.cf['model_name'] == 'xlate')):
                        outputs=outputs>0.5
                if (self.cf['model_name'] == 'xnet'):
                        outputs = (outputs[0]>0.5)
#                        print(outputs.shape)
                elif (self.cf['model_name'] == 'small') or (self.cf['model_name'] == 'unet3d'):
                    outputs = (self.model(inputs.to(self.device))>0.5)
                # statistics 
                running_dice[i] = dice_dm(((outputs).cpu().float().numpy()),labels.cpu().float().numpy())
                running_hd[i] = hd_dm(compute_surface_distances_dm((outputs[0,0,:].cpu().float().numpy()),labels[0,0,:].cpu().numpy(),spacing_mm=vox_spacing),percent=95)
                running_msd[i] = np.mean(np.array(msd(compute_surface_distances_dm((outputs[0,0,:].cpu().float().numpy()),labels[0,0,:].cpu().numpy(),spacing_mm=vox_spacing))))                  

                print('-Patient {}-: Dice={} --- HD={} --- MSD={}'.format(name[0],running_dice[i],running_hd[i],running_msd[i]))
                if self.cf['thumbnail']:
                    thumbnail(inputs[0,0,].detach().cpu().numpy(),
                              labels[0,0,].detach().cpu().numpy(),
                              outputs[0,0,].detach().cpu().float().numpy(),
                              name[0], thumb_path)
            print('Dice: %.3f'%(np.nanmean(running_dice)))
            print('HD: %.3f'%(np.mean(running_hd[np.isfinite(running_hd)])))
            print('MSD: %.3f'%(np.nanmean(running_msd)))
            print('Median Dice: %.3f'%(np.nanmedian(running_dice)))
            print('Median HD: %.3f'%(np.median(running_hd[np.isfinite(running_hd)])))
            print('Median MSD: %.3f'%(np.nanmedian(running_msd)))
            np.save(self.savepath+'/dices.npy',running_dice)
            np.save(self.savepath+'/hds.npy',running_hd)
            np.save(self.savepath+'/msds.npy',running_msd)
        print('Finished testing')
        
    def predict(self, test_gen):
        self.model.load_state_dict(torch.load(self.cf['weights_test_file']))
        if self.cf['test_set']:
            out_path=self.savepath+'/segm_results_test/'
            out_path_npy=self.savepath+'/segm_results_npy_test/'
        elif self.cf['extra_val_set']:
            out_path=self.savepath+'/segm_results_extra_validation/'
            out_path_npy=self.savepath+'/segm_results_npy_extra_validation/'
        elif self.cf['val_set']:
            out_path=self.savepath+'/segm_results_validation/'
            out_path_npy=self.savepath+'/segm_results_npy_validation/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(out_path_npy):
            os.makedirs(out_path_npy)         
        self.model.eval()
        with torch.no_grad(): 
            for i, (data, patient_name) in enumerate(test_gen, 0):
                # get the inputs         
                if (self.cf['model_name'] == 'xnet'):
                    inputs,labels, labels_box = data
                else:
                    inputs, labels = data
                if ((self.cf['model_name'] == 'ynet') or (self.cf['model_name'] == 'xnet') or (self.cf['model_name'] == 'xlate')):
                    input_a =  inputs[:,0:3] #change
                    input_b = inputs[:,3].unsqueeze(0)
    #                    print(input_a.shape, inputs.shape, input_b.shape)
                    outputs=(self.model(input_a.to(self.device),input_b.to(self.device)))
                    if ((self.cf['model_name'] == 'ynet') or (self.cf['model_name'] == 'xlate')):
                        outputs=outputs>0.5
                if (self.cf['model_name'] == 'xnet'):
                        outputs = (outputs[0]>0.5)
    #                        print(outputs.shape)
                elif (self.cf['model_name'] == 'small') or (self.cf['model_name'] == 'unet3d'):
                    outputs = (self.model(inputs.to(self.device))>0.5)
             
                for image in range(len(patient_name)):
                    result = (outputs[image,0,].detach().cpu().numpy())#.transpose(1,2,0)
                    this_input = (inputs[image,0,].detach().cpu().numpy())#.transpose(1,2,0)
                    this_label = (labels[image,0].detach().cpu().numpy())#.transpose(1,2,0)
                    
                    np.save(out_path_npy+patient_name[image]+'_out.npy',(result).astype(float))
                    np.save(out_path_npy+patient_name[image]+'_t1c_in.npy',this_input)
                    np.save(out_path_npy+patient_name[image]+'_true.npy',(this_label).astype(int))
                    nib.save(nib.Nifti1Image(this_input, np.eye(4)),out_path+patient_name[image]+'_in')
                    nib.save(nib.Nifti1Image(this_label, np.eye(4)),out_path+patient_name[image]+'_true')
                    nib.save(nib.Nifti1Image(result.astype(int), np.eye(4)),out_path+patient_name[image]+'_out')
        print('Finished predicting')
                             
    def _one_epoch(self, loader, phase):
        if phase =='train':
            thumb_path=self.savepath+'/thumb_train/' 
            thumb_path2=self.savepath+'/thumb_train_t2/' 
            thumb_path3=self.savepath+'/thumb_train_t1/' 
            thumb_path4=self.savepath+'/thumb_train_dw/' 
        else:
            thumb_path=self.savepath+'/thumb_eval/' 
            thumb_path2=self.savepath+'/thumb_eval_t2/' 
            thumb_path3=self.savepath+'/thumb_eval_t1/' 
            thumb_path4=self.savepath+'/thumb_eval_dw/' 
        if self.cf['thumbnail']:
            if not os.path.exists(thumb_path):
                os.makedirs(thumb_path)  
            if not os.path.exists(thumb_path2):
                os.makedirs(thumb_path2)    
            if not os.path.exists(thumb_path3):
                os.makedirs(thumb_path3)  
            if not os.path.exists(thumb_path4):
                os.makedirs(thumb_path4)    
        with torch.set_grad_enabled(phase=='train'):
            if phase=='train':
                self.model.train()
            else:
                self.model.eval()
            running_loss = 0.0
            running_dice = 0.0
            
            for i, (data, name) in enumerate(loader, 0):
                
                if (self.cf['model_name'] == 'xnet') or (self.cf['model_name'] == 'xnet_linear') or (self.cf['model_name'] == 'det_u') or (self.cf['model_name'] == 'xnet_mp'):
                    inputs,labels, labels_box = data
                else:
                    inputs, labels = data
                if ((self.cf['model_name'] == 'ynet') or (self.cf['model_name'] == 'xnet') or (self.cf['model_name'] == 'xnet_linear') or (self.cf['model_name'] == 'xlate')):
                    input_a =  inputs[:,0:3] #change
                    input_b = inputs[:,3].unsqueeze(0)
#                    print(input_a.shape, inputs.shape, input_b.shape)
                    outputs=(self.model(input_a.to(self.device),input_b.to(self.device)))
                # forward + backward + optimize
#                    if (self.cf['model_name'] == 'xnet') or (self.cf['model_name'] == 'xnet_linear'):
#                        # print(outputs.shape)
#                        outputs_2 = outputs[1]
#                        outputs = outputs[0]
                if ((self.cf['model_name'] == 'ynet_mp') or (self.cf['model_name'] == 'xnet_mp') or (self.cf['model_name'] == 'xlate_mp')):
                    input_a =  inputs[:,0].unsqueeze(0) 
                    input_b = inputs[:,1].unsqueeze(0)
                    input_c =  inputs[:,2].unsqueeze(0) 
                    input_d = inputs[:,3].unsqueeze(0)
#                    print(input_a.shape, inputs.shape, input_b.shape)
                    outputs=(self.model(input_a.to(self.device),input_b.to(self.device),input_c.to(self.device),input_d.to(self.device)))
                # forward + backward + optimize
                if (self.cf['model_name'] == 'xnet') or (self.cf['model_name'] == 'xnet_mp'):
                        outputs_2 = outputs[1]
                        outputs = outputs[0]
#                        print(outputs.shape)
                elif (self.cf['model_name'] == 'det_u'): #not usable anymore (changed previous elif to if)
                    outputs_coor = (self.model(inputs.to(self.device)))
                    outputs=torch.zeros_like(labels)
                    outputs[0,0,outputs_coor[0,0].int():outputs_coor[0,1].int(),outputs_coor[0,2].int():outputs_coor[0,3].int(),outputs_coor[0,4].int():outputs_coor[0,5].int()]=1
                    print(outputs_coor.int())
                elif (self.cf['model_name'] == 'small') or (self.cf['model_name'] == 'unet3d'):
                    outputs = (self.model(inputs.to(self.device)))
                if self.cf['thumbnail']:
#                    print('saving')
                    thumbnail(inputs[0,0,].detach().cpu().numpy(),
                              labels[0,0,].detach().cpu().numpy(),
                              (outputs[0,0,]>0.5).cpu().numpy(),
                              name[0], thumb_path)
                    thumbnail(inputs[0,1,].detach().cpu().numpy(),
                              labels[0,0,].detach().cpu().numpy(),
                              (outputs[0,0,]>0.5).cpu().numpy(),
                              name[0], thumb_path2)
                    thumbnail(inputs[0,2,].detach().cpu().numpy(),
                              labels[0,0,].detach().cpu().numpy(),
                              (outputs[0,0,]>0.5).cpu().numpy(),
                              name[0], thumb_path3)
                    thumbnail(inputs[0,3,].detach().cpu().numpy(),
                              labels[0,0,].detach().cpu().numpy(),
                              (outputs[0,0,]>0.5).cpu().numpy(),
                              name[0], thumb_path4)
                if (self.cf['model_name'] == 'xnet') or (self.cf['model_name'] == 'xnet_mp'):
                    loss = self.criterion(outputs.cpu(), labels.cpu().float()) + self.criterion(outputs_2.cpu(), labels_box.cpu().float())
                elif (self.cf['model_name'] == 'xnet_linear'):
                    loss = self.criterion(outputs.cpu(), labels.cpu().float()) + self.criterion2(outputs_2.cpu(), labels_box.cpu().float())/(6*112)
                elif (self.cf['model_name'] == 'det_u'):
                    loss = self.criterion2(outputs_coor.cpu(), labels_box[0,].cpu().float())/(6*112)   
                    print(loss)
                else:
                    loss = self.criterion(outputs.cpu(), labels.cpu())
                if phase=='train':
                    self.optimizer.zero_grad()               
                    loss.backward()
                    self.optimizer.step()
                dice_coef = dice((outputs>0.5).float().cpu(),labels.cpu())
                # print statistics
                running_loss += loss.item()
                running_dice += dice_coef.item()
            epoch_loss=running_loss/len(loader)#fix for more than 1 bs 
            epoch_dice=running_dice/len(loader)
        return epoch_loss, epoch_dice
    
    def _one_epoch_repeat(self, loader, phase):   
        with torch.set_grad_enabled(phase=='train'):
            if phase=='train':
                self.model.train()
            else:
                self.model.eval()
            running_dice = 0.0
            running_hd = 0.0
            if self.cf['resize']: vox_spacing = [275/112*x for x in (0.79323106, 0.7936626, 0.78976499)]
            for i, (data, name) in enumerate(loader, 0):

                inputs, labels = data

                outputs = ((self.model(inputs.to(self.device))).float()>0.5).cpu().float().numpy()
                running_dice += dice_dm((outputs),labels.cpu().float().numpy())
                running_hd += hd_dm(compute_surface_distances_dm((outputs[0,0,:]),labels[0,0,:].cpu().numpy(),spacing_mm=vox_spacing),percent=95)

                
            epoch_dice=running_dice/len(loader)
            epoch_hd=running_hd/len(loader)

        return epoch_dice, epoch_hd