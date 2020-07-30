 # -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:42:19 2020

@author: Pankaj Mishra
"""

import torch
import torch.nn as nn

# For encoding the features of res extracted feautres
class encoder(nn.Module):
    def __init__(self, in_channels):
        super(encoder, self).__init__()
        self.encoder_ = nn.Sequential(
            nn.Conv2d(in_channels =in_channels , out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine =True),
            nn.ReLU(True),            
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8,affine = True),
            nn.ReLU(True),           
            nn.Conv2d(8,8,1,stride =1),
#            nn.Ta♠nh()
        )

    def forward(self, x):
        x = self.encoder_(x)
        # x = self.decoder(x)
        return x
 
# For getting intermediate features shape equals to res net features i.e 512 features. Decoder1 is basically upsampling
class decoder1(nn.Module):
    def __init__(self, mf):
        super(decoder1, self).__init__()
        self.decoder1 = nn.Sequential(
             nn.Linear(64, 128),
             nn.BatchNorm1d(128, affine= True),
             nn.ReLU(inplace=True),
             nn.Linear(128,512),
             nn.BatchNorm1d(512, affine=True),
             nn.ReLU(inplace=True),
             nn.Linear(512,512*mf*mf),
             nn.ReLU(inplace=True))
        
    def forward(self, x):
        IF = self.decoder1(x)
        return IF
    
# For getting final reconstruction    
class decoder2(nn.Module):
    def __init__(self, in_channels):
        super(decoder2, self).__init__()
        self.decoder2 = nn.Sequential(
             nn.ConvTranspose2d(in_channels= in_channels, out_channels=16,kernel_size= 3, stride=2,padding=1),  # In b, 8, 8, 8 >> out b, 16, 10, 10
             nn.BatchNorm2d(16, affine = True),
             nn.ReLU(True),            
             nn.ConvTranspose2d(16, 32, 3, stride=2, padding = 1),  #out> b,32, 14, 14
             nn.BatchNorm2d(32, affine = True),
             nn.ReLU(True),             
             nn.ConvTranspose2d(32, 32, 4, stride=2),  #out> b, 64, 19, 19
             nn.BatchNorm2d(32, affine = True),
             nn.ReLU(True),             
             nn.ConvTranspose2d(32, 3, 4, stride=2, padding =1),  #out> b, 32, 24, 24
             nn.Tanh()
             )
        
    def forward(self, x):
         recon = self.decoder2(x)
         return recon
    
    
#### testing Code #####
if __name__ == "__main__":
    from torchsummary import summary
    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = encoder(512)
            self.Decoder1 = decoder1(1)
            self.Decoder2 = decoder2(32)
            
        def forward(self, x):
            count = x.size(0)
            y = x
            Z = self.encoder(y)
            IF = self.Decoder1(Z.view(count,-1))
            IF = torch.stack([IF,IF,IF,IF]).view(count, -1,8,8)
            Recon = self.Decoder2(IF)
            return Z, IF, Recon
    model = Autoencoder().cuda()
    print(model)
    summary(model, input_size = (512,1,1))

'''
    
if __name__ == "__main__":
    from torchsummary import summary
    model = Autoencoder().cuda()
    print(model)
    summary(model, input_size=(3,28,28))
'''