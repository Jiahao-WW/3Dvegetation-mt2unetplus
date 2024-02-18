#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :dataload.py
@说明        :
@时间        :2023/03/20
@作者        :Jiahao W
'''

from osgeo import gdal
from modelUnet import UnetPlusPlus
import numpy as np 
import math
import torch


def mask_normalization(image):
    mask = ~(np.all(image == 0, axis=0))
    u = np.mean(image, axis=(1,2), where = mask)
    sigma = np.std(image, axis=(1,2), where = mask)
    img = np.divide(image - u[:,np.newaxis, np.newaxis], sigma[:,np.newaxis, np.newaxis])
    return img, mask

if __name__ == '__main__':
    torch.cuda.empty_cache()
    inputpath = 'E:/data/GF7/beijing.tif'
    output_dir =  './data/mydataset/result/beijing.tif' 
    model_path = "./data/mydataset/model/Unetp0625_best.pth"

    #sliding window
    dx = 1024
    dy = 1024
    #overlap
    overlap = 256
    #channel
    channels = 4
    #class
    classes = 2
    #prediction
    half = overlap//2 
    data_set = gdal.Open(inputpath)
    img = data_set.ReadAsArray()

    img = np.array(img, np.float32)/10000.0
    '''NDVI = (img[3]-img[2])/(img[3]+img[2])
    NDVI = np.nan_to_num(NDVI)
    NDVI = np.expand_dims(NDVI, axis=0)'''
    #img = np.concatenate((img,NDVI),axis=0)

    img, mask = mask_normalization(img)
    bands, m,n = img.shape

    numx = math.ceil((m - overlap)/(dx - overlap))
    numy = math.ceil((n - overlap)/(dy - overlap))

    print('load model')

    model = UnetPlusPlus(
    encoder_name="resnext50_32x4d",   
    encoder_weights="imagenet",     
    in_channels=channels,   
    classes=classes,
    )
 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if torch.cuda.device_count()>1:
        print('GPUs')
        model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    #initialization
    outputseg = np.zeros((m,n), dtype='float')
    outputreg = np.zeros((m,n), dtype='float')

    t = 0
    for i in range(numx):
        x = (i+1)*dx - i*overlap
         #whether the sliding window reaches the edge of the image
        if x>m :
            x = m
        for j in range(numy):
            y = (j+1)*dy - j*overlap
             #whether the sliding window reaches the edge of the image
            if y>n :
                y = n
            temp = img[:, x-dx:x, y-dy:y]
            temp = np.expand_dims(temp, axis=0)
            temp = torch.Tensor(temp)
            temp = temp.cuda()
            segout, regout = model(temp)
            segout = torch.argmax(segout, axis=1)
            segout = segout.squeeze(1)
            regout = regout.squeeze(1)
            segout = segout.squeeze(0)
            regout = regout.squeeze(0)
            regout = regout.cpu().data.numpy()
            segout = segout.cpu().data.numpy()
            outputseg[x-dx+half:x-half, y-dy+half:y-half] = segout[half:dx-half,half:dy-half]
            outputreg[x-dx+half:x-half, y-dy+half:y-half] = regout[half:dx-half,half:dy-half]
            t = t+1
            print('finished:' + '%d'%t)

    m,n = outputreg.shape
    datatype = gdal.GDT_Float32
    height = outputseg*outputreg

    img_tran = data_set.GetGeoTransform()
    img_proj = data_set.GetProjection()
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_dir, n, m, 1,datatype)
    dataset.SetGeoTransform(img_tran)   # affine transformation
    dataset.SetProjection(img_proj)     # projection
    dataset.GetRasterBand(1).WriteArray(height)
    print('finish')
    del dataset

