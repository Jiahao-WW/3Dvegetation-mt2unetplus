#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :dataload.py
@说明        :
@时间        :2023/03/01
@作者        :Jiahao W
'''

import torch 
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import numpy as np
import os
from torchvision import transforms
from osgeo import gdal
import random
import numpy as np
from skimage.transform import  rotate
from skimage.exposure import rescale_intensity


#normalize
transform = transforms.Compose([
    transforms.ToTensor()])

def randomtransforms(image, label):
    # random rotation
    if random.random() > 0.5:
        angle = np.random.randint(4) * 90
        image = rotate(image, angle).copy() # 1: Bi-linear (default)
        label = rotate(label, angle, order=0).copy() # Nearest-neighbor
    
    # flip left and right
    if random.random() > 0.5:
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()

    # flip up and down
    if random.random() > 0.5:
        image = np.flipud(image).copy()
        label = np.flipud(label).copy()

    # brightness
    ratio=random.random()
    if  ratio>0.5:
        image = rescale_intensity(image, out_range=(0, ratio)).copy() #(0.5, 1)

    return image, label


def normalize(image):
    img = np.zeros(image.shape, dtype='float32')
    u = np.mean(image, axis=(1,2))
    sigma = np.std(image, axis=(1,2))
    img = (image - u[:,np.newaxis, np.newaxis])/ sigma[:,np.newaxis, np.newaxis]
    return img

def read_data(root_path, mode = 'train'):
    images = []
    masks = []
 
    image_root = os.path.join(root_path, mode + '\img')
    gt_root = os.path.join(root_path, mode + '\label')
 
 
    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        images.append(image_path)
    
    for gt_name in os.listdir(gt_root):
        label_path = os.path.join(gt_root, gt_name)
        masks.append(label_path)
    return images, masks

def data_loader(img_path, mask_path):
    #read image
    img_path = os.path.abspath(img_path)
    img = gdal.Open(img_path).ReadAsArray()
    img = np.array(img, np.float32)
    img = img/10000.0

    # NDVI
    '''NDVI = (img[3]-img[2])/(img[3]+img[2])
    NDVI = np.nan_to_num(NDVI)
    NDVI = np.expand_dims(NDVI, axis=0)'''

    # NDWI
    '''NDWI = (img[3]-img[1])/(img[3]+img[1])
    NDWI = np.nan_to_num(NDWI)
    NDWI = np.expand_dims(NDWI, axis=0)'''

    #EVI
    '''EVI = 2.5*(img[3]-img[2]) / (img[3]+6*img[2]-7.5*img[0]+1)
    EVI = np.nan_to_num(EVI)
    EVI = np.expand_dims(EVI, axis=0)'''

    #comimg = np.concatenate((img,NDVI), axis=0)
    #comimg = np.concatenate((img,EVI), axis=0)
    #comimg = np.concatenate((comimg,NDWI), axis=0)

    comimg = normalize(img) #normalization
    comimg = comimg.transpose(1, 2, 0)
    mask = gdal.Open(mask_path).ReadAsArray()
    mask = np.array(mask, np.float32) 
    
    comimg, mask = randomtransforms(comimg, mask) # data enhancement

    mask = np.expand_dims(mask, axis=0) # add axis
    comimg = comimg.transpose(2, 0, 1)

    return comimg, mask

def data_test_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    mask = np.expand_dims(mask, axis=2)

  
    img = np.array(img, np.float32) 
    mask = np.array(mask, np.float32)

    return img, mask

class Mydataset(data.Dataset):

    def __init__(self, rootpath, mode='train'):
        self.root = rootpath
        self.mode = mode
        self.images, self.labels = read_data(self.root, self.mode)

    def __getitem__(self, index):
        if self.mode == 'test':
            img, mask = data_test_loader(self.images[index], self.labels[index])
        else:
            img, mask = data_loader(self.images[index], self.labels[index])
            img = torch.Tensor(img)
            mask2 = np.zeros_like(mask, dtype = np.float32)
            mask2[mask > 0] = 1 
            mask = torch.Tensor(mask) # regression label
            mask2 = torch.Tensor(mask2) #segmentation label
        return img, mask2, mask

    def __len__(self):
        assert len(self.images) == len(self.labels) # The number of images and labels must be equal
        return len(self.images)
