# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:11:28 2020

@author: Pankaj Mishra
"""

import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg11(pretrained=True).features[:4].eval())
        #### IF you want o use deeper layer for the perception (uncomment below code) ####
#        blocks.append(torchvision.models.vgg11(pretrained=True).features[4:9].eval()) 
#        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
#        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:22].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y,reduction = 'mean')
        return loss

## Testing ####   
if __name__=='__main__':
     loss = VGGPerceptualLoss().cuda()
     x = torch.randn(2,3,120,120).cuda()
     y = torch.randn(2,3,120,120).cuda()
     l = loss(x,y)
     print(l.item())