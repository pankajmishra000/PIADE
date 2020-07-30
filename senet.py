# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:55:07 2020

@author: Pankaj Mishra

Code adapted from - https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
"""

from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
#        print(f' bacth{b} channel {c}')
        y = self.avg_pool(x).squeeze(2).squeeze(2)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)