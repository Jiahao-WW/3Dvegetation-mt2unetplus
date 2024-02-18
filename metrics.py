#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :metrics.py
@说明        :
@时间        :2023/03/12
@作者        :Jiahao W
'''


import torch
import sklearn

# Confusion matrix
def cal_cm(y_true, y_pred):
    y_true = y_true.reshape(1, -1).squeeze() 
    y_pred = y_pred.reshape(1, -1).squeeze()
    cm = sklearn.metrices.condusion_matrix(y_true, y_pred)
    return cm

# accuracy calculation
def ACC(pred, label):
    corrects = torch.eq(pred, label).int()
    acc = corrects.sum()/corrects.numel()
    return acc

def RMSE(pred, label):
    sub = torch.sub(pred,label)
    rmse = torch.sqrt(sub.square().sum()/sub.numel())
    return rmse


def MAE(pred, label):
    sub = torch.sub(pred,label)
    mae = sub.abs().sum()/sub.numel()
    return mae

def IOU(y_pred, y_true):

    y_pred = y_pred.float()
    y_true = y_true.float()

    #  TP、FP 和 FN
    TP = (y_pred * y_true).sum()
    FP = ((1 - y_true) * y_pred).sum()
    FN = (y_true * (1 - y_pred)).sum()

    #  IoU
    iou = TP / (TP + FP + FN + 1e-6)
    return iou