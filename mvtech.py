# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 18:47:39 2020

@author: Pankaj Mishra
"""
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from collections import OrderedDict

def read_files(root,d, product, data_motive = 'train', use_good = True, normal = True):
    '''
    return the path of the train directory and list of train images
    
    Parameters:
        root : root directory of mvtech images
        d = List of directories in the root directory
        product : name of the product to return the images for single class training.Products are-
            ['all','bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 
            'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        data_motve : Can be 'train' or 'test' based on the intention of the data loader function
        use_good : To use the data in the good folder. For training the default is False as we need the data of good folder.
        normal : Signofy if the normal imgaes are included while loading or not. Accepts boolean value  True or False
        
    Returns:
        Path and Image ordered dict for the test dataset
    '''
    files = os.listdir(os.path.join(root,d))
        #    print(files)
    for d_in in files:
        if os.path.isdir(os.path.join(root,d,d_in)):
            if d_in == data_motive:
                im_pt = OrderedDict()
                file = os.listdir(os.path.join(root,d, d_in))
                
                for i in file:
                    if os.path.isdir(os.path.join(root, d, d_in,i)):
                        if (data_motive == 'train'):
                            tr_img_pth = os.path.join(root, d, d_in,i)
                            images = os.listdir(tr_img_pth)
                            im_pt[tr_img_pth] = images
                            print(f'total {d_in} images of {i} {d} are: {len(images)}')
                            
                        if (data_motive == 'test') :
                            if (use_good == False) and (i == 'good') and normal != True:
                                print(f'the good images for {d_in} images of {i} {d} is not included in the test anomolous data')
                            elif (use_good == False) and (i != 'good') and normal != True :
                                tr_img_pth = os.path.join(root, d, d_in,i)
                                images = os.listdir(tr_img_pth)
                                im_pt[tr_img_pth] = images
                                print(f'total {d_in} images of {i} {d} are: {len(images)}')
                            elif (use_good == True) and (i == 'good') and (normal== True):
                                tr_img_pth = os.path.join(root, d, d_in,i)
                                images = os.listdir(tr_img_pth)
                                im_pt[tr_img_pth] = images
                                print(f'total {d_in} images of {i} {d} are: {len(images)}')                            
                if product == "all":
                    return
                else:
                    return im_pt #tr_img_pth, images
                    
def load_images(path, image_name):
    return imread(os.path.join(path,image_name))

    
def Test_anom_data(root, product= 'bottle', use_good = False):
    '''
    return the path of the train directory and list of train images
    
    Parameters:
        root : root directory of mvtech images
        product : name of the product to return the images for single class training.Products are-
            ['all','bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 
            'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        use_good : To use the data in the good folder. For training the default is False as we need the data of good folder.
        
    Returns:
        Path and Image ordered dict for the test dataset
    '''
    dir = os.listdir(root)
    
    for d in dir:
        if product == "all":
            read_files(root, d, product, data_motive = 'test',use_good = use_good,normal = False)
            
        elif product == d:
            pth_img_dict= read_files(root, d, product,data_motive='test', use_good = use_good, normal = False)
            return pth_img_dict
        

def Test_normal_data(root, product= 'bottle', use_good = True):
    if product == 'all':
        print('Please choose a valid product. Normal test data can be seen product wise')
        return
    dir = os.listdir(root)
    
    for d in dir:
        if product == d:
            pth_img = read_files(root, d, product,data_motive='test',use_good = True, normal = True)
            return pth_img
    
                      
def Train_data(root, product = 'bottle', use_good = True):
    '''
    return the path of the train directory and list of train images
    
    Parameters:
        root : root directory of mvtech images
        product : name of the product to return the images for single class training.Products are-
            ['all','bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 
            'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        use_good : To use the data in the good folder. For training the default is True as we need the data of good folder.
        
    Returns:
        Path and Image ordered dict for the training dataset
    '''
    
    dir = os.listdir(root)
    
    for d in dir:
        if product == "all":
            read_files(root, d, product,data_motive='train')            

        elif product == d:
            pth_img = read_files(root, d, product,data_motive='train')
            return pth_img
        
        
class Mvtec:
    def __init__(self, batch_size,root="D:\\2ND YEAR\\MVT_Anom_dataset\\mvtec_anomaly_detection", product= 'bottle'):
        self.root = root
        self.batch = batch_size
        self.product = product
        torch.manual_seed(123)
        
        if self.product == 'all':
            print('--------Please select a valid product.......See Train_data function-----------')
        else:
         # Importing all the image_path dictionaries for  test and train data #
            train_path_images =Train_data(root = self.root, product = self.product)
            test_anom_path_images = Test_anom_data(root = self.root, product=self.product)
            test_norm_path_images = Test_normal_data(root= self.root, product = self.product)
            
            ## Image Transformation ##
            T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((156,156)),
                transforms.CenterCrop(120),
#                transforms.Resize(120),
                transforms.ToTensor(),
    #            transforms.Normalize((0.1307,), (0.3081,)),
            ])
            train_normal = torch.stack([T(load_images(j,i)) for j in train_path_images.keys() for i in train_path_images[j]])
            test_anom = torch.stack([T(load_images(j,i)) for j in test_anom_path_images.keys() for i in test_anom_path_images[j]])
            test_normal = torch.stack([T(load_images(j,i)) for j in test_norm_path_images.keys() for i in test_norm_path_images[j]])
            print(f' --Size of {self.product} train loader: {train_normal.size()}--')
            print(f' --Size of {self.product} test anomaly loader: {test_anom.size()}--')
            print(f' --Size of {self.product} test normal loader: {test_normal.size()}--')          
            
            ####  Final Data Loader ####
            self.train_loader  = torch.utils.data.DataLoader(train_normal, batch_size=batch_size, shuffle=True)            
            self.test_anom_loader = torch.utils.data.DataLoader(test_anom, batch_size = batch_size, shuffle=False)
            self.test_norm_loader = torch.utils.data.DataLoader(test_normal, batch_size=batch_size, shuffle=False)
            
        
            
            
if __name__ == "__main__":
    
    root = "D:\\2ND YEAR\\MVT_Anom_dataset\\mvtec_anomaly_detection"
    print('======== All Normal Data ============')
    Train_data(root, 'all')
    print('======== All Anomaly Data ============')
    Test_anom_data(root,'all')    
          
    train = Mvtec(1,root,'bottle')
    preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((156,156)),
            transforms.CenterCrop(120),
            transforms.ToTensor(),])
    for i in train.test_anom_loader:
        i = preprocess(i[0])
        print(i.shape)
        plt.imshow(i.squeeze(0).permute(1,2,0))
        plt.show()
        break
    
        
                           
                            
                
