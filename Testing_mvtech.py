# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:24:25 2020

@author: Pankaj Mishra
"""


from config import Config
import torch
import torch.nn.functional as F
import torchvision
from capsmod import PSPNet
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mvtech
from skimage.measure import compare_ssim
import numpy as np
import torchvision.utils as utils
import pytorch_ssim
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import vgg_percep_loss
import time

def plot(img, reconstruction, ssim):
    plt.subplot(231)
    plt.imshow(img[0].permute(1,2,0).detach().cpu().numpy())
    #plt.xlabel(f'SSIM: {ssim[0]:.2f}')
    plt.subplot(232)
    plt.imshow(img[1].permute(1,2,0).detach().cpu().numpy())
    #plt.xlabel(f'SSIM: {ssim[1]:.2f}')
    plt.subplot(233)
    plt.imshow(img[2].permute(1,2,0).detach().cpu().numpy())
    #plt.xlabel(f'SSIM: {ssim[2]:.2f}')
    
    plt.subplot(234)
    plt.imshow(reconstruction[0].permute(1,2,0).detach().cpu().numpy())
    plt.xlabel(f'SSIM: {ssim[0]:.2f}')
    plt.subplot(235)
    plt.imshow(reconstruction[1].permute(1,2,0).detach().cpu().numpy())
    plt.xlabel(f'SSIM: {ssim[1]:.2f}')
    plt.subplot(236)
    plt.imshow(reconstruction[2].permute(1,2,0).detach().cpu().numpy())
    plt.xlabel(f'SSIM: {ssim[2]:.2f}')
    plt.show()
    return print('all printed')

batch_size = 3

config = Config()

model = PSPNet(test = True)
prd = ['bottle']#,'capsule','carpet']#, 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
#            'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

for norm_class in prd:
     t=open(f'score.txt','a')
     t.write(f'#################### CLASS {norm_class}  ####################\n')
     if config.USE_CUDA:
         model = model.cuda()
     model.load_state_dict(torch.load(f'Mvtech_{norm_class}'+'.pt'))
#     model.load_state_dict(torch.load(f'./ICPR2020-main results/Mvtech_metal_nut1routing_4_mag_2_caps_ssim_'+'.pt'))
     model.eval()
     
     data = mvtech.Mvtec(batch_size, product=norm_class)
     
            
     ### Coefficient of losses ###
     def min_max_coefficient(lst):
         lss = np.array(lst)
         return lss.min(), lss.max()
     
     ### Novel Anomaly Score Function ####
     def anomaly_Score(l_recon, l_pf, l_pi, recon_min,recon_max, l_pf_min, l_pf_max, l_pi_min, l_pi_max):
         '''Returns the novel anomaly score for a new sample over the trained model.
         
         Arguments:
             recon_min : Min value of the reconstion over the held out set.
             recon_max : Max value of the reconstion over the held out set.
             lll_min   : Min value of the log likelihood loss over the held out set.
             lll_max   : Max value of the reconstion over the held out set.
             
         Input:
             l_recon : Reconstruction loss of new sample.
             l_lll   : Log likelihood of new sample
             
         Output:
             score : normalised anomaly score
         '''
         norm_recons = (l_recon - recon_max)/(recon_max - recon_min)
         norm_l_pf = (l_pf - l_pf_min)/(l_pf_max - l_pf_min)
         norm_l_pi = (l_pi - l_pi_min)/(l_pi_max - l_pi_min)
         score = norm_recons + norm_l_pf + norm_l_pi
         return score
     ### Perceptual and SSIM loss #####

     ssim_loss = pytorch_ssim.SSIM()  # SSIM Loss
     perc_loss = vgg_percep_loss.VGGPerceptualLoss(resize=False).cuda()
     
     def show(img):
         npimg = img.cpu().detach().numpy()
         plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
         plt.show(block=False)
#         plt.pause(2)
#         plt.cla()
         plt.close()
         

         ########## Testing #############
     loader = [data.test_norm_loader,data.test_anom_loader] #
     name = ['normal data', 'anom data']
     for n,load in enumerate(loader):
         loss1_ = []
         loss2_ = []
         loss3_ = []
         loss4_ = []
         loss_ = []
         IF_ = []
         with torch.no_grad():
             for no ,img in enumerate(load):
                 if no < 5:
                     if img.size(1)==1:
                         img =torch.stack([img,img,img]).squeeze(2).permute(1,0,2,3) 
     #                img =torch.stack([img,img,img]).squeeze(2).permute(1,0,2,3)
                     if config.USE_CUDA:
                         img = img.cuda()
                     show(utils.make_grid(img, nrow = 5))
                     Cstart_time = time.time()
                     reconstruction,res, IF, vectors = model(img)
                     start_time = time.time()
                     print(f'Time taken for inference: {time.time() - start_time} seconds')
                     vec_norms = torch.norm(vectors, dim=2)
                     IF_.append(IF.detach().cpu().numpy())
                     show(utils.make_grid(reconstruction, nrow = 5))
                 
                     img_np = img.detach().cpu().permute(0,2,3,1).numpy()
                     recon_img_np = reconstruction.detach().cpu().permute(0,2,3,1).numpy()
                     ssim = []
                 
                     for i in range(img_np.shape[0]):
                         loss1 = F.mse_loss(reconstruction[i], img[i], reduction='mean')
                         loss2 = -ssim_loss(img[i].unsqueeze(0),reconstruction[i].unsqueeze(0) )
                         # loss3 = -torch.sub(vec_norms[:,0], vec_norms[:,1])[i]
                         loss4 = perc_loss(reconstruction[i].unsqueeze(0),img[i].unsqueeze(0))
                         # loss4 = F.mse_loss(perc_loss1(reconstruction), perc_loss1(img), reduction='mean')
                         ssim.append(pytorch_ssim.ssim(img[i].unsqueeze(0),reconstruction[i].unsqueeze(0)).item())
                         loss1_.append(loss1.item())
                         loss2_.append(loss2.item())
                         # loss3_.append(loss3.item())
                         loss4_.append(loss4.item())
                         loss = loss1 + loss2 + loss4
                         loss_.append(loss.item())
                     print(f'ssim values test {name[n]} batch {no} data : {ssim}')
                     plot(img, reconstruction, ssim)
     
             # Plotting individual losses
             plt.plot(np.array(loss1_), 'r.', label = "recons loss", alpha = 0.9 )
             plt.plot(np.array(loss2_), 'b+', label = "SSIM loss",alpha = 0.5)
             plt.plot(np.array(loss3_), 'y^', label = "Vector length loss",alpha = 0.3)
             plt.plot(np.array(loss4_), 'g*', label="perceptual loss", alpha=0.3)
             plt.plot(np.array(loss_), 'p-', label="total loss", alpha=0.3)
             plt.title(f"Loss 1 Loss3, Loss4 and total loss for {name[n]}")
             plt.legend()
             plt.show(block=False)
             plt.savefig(f'{norm_class}_Losses_{name[n]}.png', dpi=300)
             plt.pause(2)
             plt.close()
             # Anomaly score
             if n ==0:
                 total_loss_normal = loss_
                 ssim_normal = ssim
                 IF_normal = IF_

             if n ==1:
                 total_loss_anomaly = loss_
                 ssim_anomlay =ssim
                 IF_anomaly = IF_
        
         
     # Plotting total losses
     plt.subplot(1,1,1)
     plt.plot(total_loss_normal, 'r-', label = "Normal total loss", alpha = 0.9 )
     plt.plot(total_loss_anomaly, 'b-', label = "Anomaly total loss" ,alpha = 0.5)
     plt.xlabel('Number of images')
     plt.ylabel('Total losses')
     plt.title("Total losses")
     plt.legend()
     plt.savefig(f'class_{norm_class}_total_losses.png',dpi=300)
     plt.show(block=False)
     plt.pause(2)
     plt.cla()
     
     ### ROC curve plotting  ###
     roc_data = np.concatenate((total_loss_normal, total_loss_anomaly))
     roc_targets = np.concatenate((np.zeros(len(total_loss_normal)), np.ones(len(total_loss_anomaly))))
     fpr, tpr, thresholds = roc_curve( roc_targets, roc_data )
     t.write(f'fpr: {fpr}\ntpr: {tpr}\nthresholds: {thresholds}\nroc AUC score: {roc_auc_score(roc_targets, roc_data)}\n')
     plt.figure()
     lw=2
     plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (score=%0.4f)'%auc(fpr,tpr))
     plt.plot([0,1],[0,1], color='navy',lw=lw,linestyle='--')
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title(f'ROC_AUC Score: {roc_auc_score(roc_targets, roc_data)}')
     plt.legend(loc='lower right')
     plt.savefig(f'class_{norm_class}_ROC_AUC_score.png',dpi=300)
     plt.show(block=False)
     plt.pause(2)
     plt.cla()
     
     ### Precision Recall Curve ###
     precision, recall, thres = precision_recall_curve(roc_targets, roc_data)
     plt.plot(recall, precision, marker='.', label='P-R Plot (score=%0.4f)'%auc(recall, precision))
     plt.xlabel('Recall')
     plt.ylabel('Precision')
     plt.title(f'AUC Score: {auc(recall, precision)}')
     t.write(f'AUC score precision recall: {auc(recall, precision)}\n')
     plt.legend(loc='lower center')
     plt.savefig(f'class_{norm_class}_AUC_score.png',dpi=300)
     plt.show(block=False)
     plt.pause(2)
     plt.cla()
     
     # compute best classification
     idx = np.argmax(tpr-fpr)
     best_thresh = thresholds[idx]
     err = ((roc_data > best_thresh) != roc_targets).sum()
     accuracy =  1 - err/roc_data.shape[0]
     print("Accuracy: ", accuracy)
     t.write(f'Accuracy: {accuracy}\n')
     t.write(f'SSIM Normal Average: {sum(ssim_normal) / len(ssim_normal)}\n')
     t.write(f'SSIM Anomaly Average: {sum(ssim_anomlay) / len(ssim_anomlay)}\n')
     t.write(f'TPR to best threshold: {tpr[idx]}\n')
     t.write(f'FPR to best threshold: {fpr[idx]}\n\n')
     t.close()
     torch.cuda.empty_cache()
     
