# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:05:16 2019

@author: Pankaj Mishra
"""


import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from config import Config
import config
from autoencoder import decoder1, decoder2
import senet

class DigitCaps(nn.Module):
    def __init__(self, out_num_caps=2, in_num_caps=32 * 2 * 2, in_dim_caps=8, out_dim_caps=64, decode_idx=-1):
        super(DigitCaps, self).__init__()

        self.conf = Config()
        self.in_dim_caps = in_dim_caps
        self.in_num_caps = in_num_caps
        self.out_dim_caps = out_dim_caps
        self.out_num_caps = out_num_caps
        self.decode_idx = decode_idx
        self.W = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
#        self.upsample = upsampling()

    def forward(self, x):
        # x size: batch x 1152 x 8
        x_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        x_hat_detached = x_hat.detach()
        # x_hat size: batch x ndigits x 1152 x 16
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps))
        # b size: batch x ndigits x 1152
        if self.conf.USE_CUDA:
            b = b.cuda()

        # routing algo taken from https://github.com/XifengGuo/CapsNet-Pytorch/blob/master/capsulelayers.py
        num_iters = 3
        for i in range(num_iters):
            c = F.softmax(b, dim=1)
            # c size: batch x ndigits x 1152
            if i == num_iters -1:
                # output size: batch x ndigits x 1 x 16
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else:
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)


        outputs = torch.squeeze(outputs, dim=-2) # squeezing to remove ones at the dimension -1
        # Below code chooses the maximum lenth of the vector
        if self.decode_idx == -1:  # choose the longest vector as the one to decode
            classes = torch.sqrt((outputs ** 2).sum(2))
            classes = F.softmax(classes, dim=1)
            _, max_length_indices = classes.max(dim=1)
        else:  # always choose the same digitcaps
            max_length_indices = torch.ones(outputs.size(0)).long() * self.decode_idx
            if self.conf.USE_CUDA:
                max_length_indices = max_length_indices.cuda()

        masked = Variable(torch.sparse.torch.eye(self.out_num_caps))
        if self.conf.USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices)
        t = (outputs * masked[:, :, None]).sum(dim=1).unsqueeze(1)


        return t, outputs

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class PyramidPool(nn.Module):

    def __init__(self, in_features, pool_size):
        super(PyramidPool, self).__init__()

        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // pool_size, 1, bias=False),
            nn.BatchNorm2d(in_features // pool_size, momentum=.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.features(x)
        return output


class PSPNet(nn.Module):

    def __init__(self, num_channels=3, pretrained = True, test = False):# change no. of classes as we are doing segmentation. We just need values for the recons.
        super(PSPNet,self).__init__()
        print("initializing model")
        self.resnet = torchvision.models.resnet50(pretrained = pretrained)

        self.layer5a = PyramidPool(2048, 1)
        self.layer5b = PyramidPool(2048, 2)
        self.layer5c = PyramidPool(2048, 4)
        self.layer5d = PyramidPool(2048, 8)
        
        self.D1 = DigitCaps(in_num_caps=480)
        # self.D2 = DigitCaps(in_num_caps=128)

        self.if_ = decoder1(4)
        self.final = decoder2(640)

        self.test = test

        ### Ateention Layer ###
        self.se1 = senet.SELayer(channel=64, reduction=16)
        self.se2 = senet.SELayer(channel=256, reduction=16)
        self.se3 = senet.SELayer(channel=512, reduction=16)
        self.se4 = senet.SELayer(channel=1024, reduction=16)
        self.se5 = senet.SELayer(channel=2048, reduction=16)

        if self.test == 'False':
            initialize_weights(self.layer5a, self.layer5b, self.layer5c, self.layer5d, self.final,
                               self.se1, self.se2, self.se3, self.se4, self.se5)


    def forward(self, x):
        count = x.size(0)
        size = x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)  # 64,60,60
        x = self.resnet.maxpool(x)
        x = self.se1(x)
        x = self.resnet.layer1(x)
        x = self.se2(x)
        x = self.resnet.layer2(x)
        x = self.se3(x)
        x = self.resnet.layer3(x)
        x = self.se4(x)
        x = self.resnet.layer4(x)
        x = self.se5(x)
        
        p1 = self.layer5a(x).view(count, -1,8)
       
        p2 = self.layer5b(x).view(count, -1,8)
        
        p3 = self.layer5c(x).view(count, -1,8)

        p4 = self.layer5d(x).view(count, -1,8)

        # concatenating the 4 vectors from D1, D2, D3, D4

        concat  = torch.cat((p1,p2,p3,p4), dim =1)
        IF1, vectors = self.D1(concat)

        IF = self.if_(IF1.squeeze(1)).view(count,-1,4,4)

        IF_ = torch.cat((x, IF), dim=1)

        recons = self.final(IF_.view(count, -1, 8, 8))

        return recons, x, IF, vectors

    
# Initialize weight function
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
##### Testing  ####

if __name__ == "__main__":
    from torchsummary import summary
    model = PSPNet()
    if config.Config:
        model = model.cuda()
    print(model)
    summary(model, input_size = (3,120,120))
    




