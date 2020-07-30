# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:54:19 2020

@author: Pankaj Mishra
."""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.utils as utils
import capsmod
import numpy as np
import vgg_percep_loss
from config import Config
import matplotlib.pyplot as plt
import pytorch_ssim
import mvtech
import os

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
conf = Config()
n_epochs =400
item = ['bottle']

for prdt in item:
    
    data = mvtech.Mvtec(60, product = prdt)
        
    ### Perceptual and SSIM loss #####

    perc_loss = vgg_percep_loss.VGGPerceptualLoss(resize=False).cuda()

    ssim_loss = pytorch_ssim.SSIM() # SSIM Loss


    #########  Training  saving the model #############
    model = capsmod.PSPNet()
    if conf.USE_CUDA:
        model = model.cuda()
    model.train()    
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    
    for epoch in range(n_epochs):
        loss_all = []
        
        for img in data.train_loader:
            if img.size(1)==1:
                img =torch.stack([img,img,img]).squeeze(2).permute(1,0,2,3)
            img = img.cuda() 
            
            model.zero_grad()
            reconstruction, res, IF, vectors = model(img) #dimension of latent (batch,32)
    
            loss1 = F.mse_loss(reconstruction, img, reduction = 'mean')
            loss2 = -ssim_loss(img,reconstruction) #-ve as we want to increase the SSIM value            
            loss3 = perc_loss(reconstruction,img)
            
            loss = loss1 + 1.0*loss2 + 1.0*loss3
            loss_all.append(loss.item())
            writer.add_scalar('Reconstruction Loss', loss1, epoch)
            writer.add_scalar('SSIM Loss', loss2, epoch)
            writer.add_scalar('IMAGE perceptual Loss', loss3, epoch)
           
            writer.add_image('Reconstructed Image',utils.make_grid(reconstruction),epoch,dataformats = 'CHW')
            
            loss.backward()        
            optimizer.step()
    
        writer.add_scalar('Mean Epoch loss', np.mean(loss_all), epoch)
        print(f">>Loss of the epoch {epoch} is: {np.mean(loss_all)}<<")
        writer.close()
        
    torch.save(model.state_dict(), f'Mvtech_{prdt}'+'.pt')
    torch.cuda.empty_cache()
    
