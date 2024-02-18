#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :train.py
@说明        :
@时间        :2023/03/12
@作者        :Jiahao W
'''

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_toolbelt import losses as L
import numpy as np
from tqdm import tqdm
from modelUnet import UnetPlusPlus
import warnings
from dataload import Mydataset
from metrics import ACC, RMSE, MAE

torch.cuda.empty_cache()

def loss_weight(seg, reg, target1, target2, log_vars):

    #seg
    W1 = torch.exp(-log_vars[0])
    diff1 = F.cross_entropy(seg, target1)
    loss1 = diff1*W1 + log_vars[0]

    #reg
    W2 = torch.exp(-log_vars[1])
    diff2 = F.mse_loss(reg, target2, reduction='mean')
    loss2 = diff2*W2*0.5 + log_vars[1]
    return loss1 + loss2, W1, W2

def loss_weight_ave(seg, reg, target1, target2):

    diff1 = F.cross_entropy(seg, target1, reduction='mean')
    #diff1 = F.binary_cross_entropy(seg, target1, reduction='mean')
    loss1 = diff1
    
    diff2 = F.mse_loss(reg, target2, reduction='mean')
    loss2 = diff2
    return loss1 + loss2

def train(EPOCHES, BATCH_SIZE, data_root,channels, n_classes,
          model_path, early_stop):
    train_dataset = Mydataset(data_root, mode = 'train')
    val_dataset = Mydataset(data_root, mode = 'val')
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.enable = True

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 0)

    val_data_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size = BATCH_SIZE,
		shuffle=True,
		num_workers=  0)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UnetPlusPlus(
	encoder_name="resnext50_32x4d",      
	encoder_weights= 'imagenet',    
	in_channels=channels,             
	classes=n_classes,         
	)

    if torch.cuda.device_count()>1:
        print('GPUs')
        model = torch.nn.DataParallel(model)
    model.to(DEVICE)

    #ciggma
    log_var_a = torch.zeros((1,), requires_grad=True)
    log_var_b = torch.zeros((1,), requires_grad=True)

    params = ([p for p in model.parameters()] + [log_var_a] + [log_var_b])

    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = 2,
        T_mult = 2,
        eta_min = 1e-5)

    #loss_fn = nn.MSELoss().cuda()
    #loss_fn = nn.BCELoss().cuda()
    #loss_fn = SoftCrossEntropyLoss_fn.cuda()
    #loss_fn = JointLoss().cuda()

    best_val = 10
    best_valmae = 10
    train_rmse_epochs,train_acc_epochs,val_rmse_epochs, val_acc_epochs, val_loss= [], [], [], [], []

    for epoch in range(1, EPOCHES+1):
        losses = []
        tra_rmse= []
        tra_acc = []
        model.train()
        for image, target1, target2 in tqdm(train_data_loader, ncols = 50, total = len(train_data_loader)):
            image, target1, target2  = image.to(DEVICE), target1.to(DEVICE),target2.to(DEVICE)
            seg, reg = model(image)

            target1 = torch.tensor(target1, dtype = torch.int64)
            target2 = torch.tensor(target2)
            target1 = target1.squeeze(1)

            loss, _, _ = loss_weight(seg, reg, target1, target2,[log_var_a.to(DEVICE), log_var_b.to(DEVICE)])
            #loss= loss_weight_ave(seg, reg, target1, target2)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #training loss
            target2 = target2.squeeze(1)
            seg = torch.argmax(seg, axis=1)
            seg = seg.squeeze(1) #4D to 3D
            reg = reg.squeeze(1)

            target = seg * reg
            acc = ACC(seg,target1)
            rmse = RMSE(target,target2)
            tra_acc.append(acc.item())
            tra_rmse.append(rmse.item())
        
        scheduler.step()

        val_rmse, val_acc, val_mae= [],[],[]
        W1 = 0
        W2 = 0
        val_data_loader_num = iter(val_data_loader)
        
        for val_img, val_mask1, val_mask2 in tqdm(val_data_loader_num, ncols = 50, total = len(val_data_loader_num)):
            vallosses = []
            val_img, val_mask1, val_mask2 = val_img.to(DEVICE), val_mask1.to(DEVICE), val_mask2.to(DEVICE)
            predict1, predict2 = model(val_img)

            val_mask1 = torch.tensor(val_mask1, dtype = torch.int64)
            val_mask2 = torch.tensor(val_mask2)
            val_mask1 = val_mask1.squeeze(1)

            valloss, W1, W2 = loss_weight(predict1, predict2, val_mask1, val_mask2, [log_var_a.to(DEVICE), log_var_b.to(DEVICE)])
            #valloss = loss_weight_ave(predict1, predict2, val_mask1, val_mask2)

            vallosses.append(valloss.item())
            val_mask2 = val_mask2.squeeze(1)
            predict1 = torch.argmax(predict1, axis=1)
            predict1 = predict1.squeeze(1) #4D to 3D
            predict2 = predict2.squeeze(1)

            target = predict1 * predict2
            acc = ACC(predict1,val_mask1)
            rmse = RMSE(target,val_mask2)
            mae = MAE(target,val_mask2)
            val_acc.append(acc.item())
            val_rmse.append(rmse.item())
            val_mae.append(mae.item())

        train_rmse_epochs.append(np.array(tra_rmse).mean())
        train_acc_epochs.append(np.array(tra_acc).mean())
        val_loss.append(np.array(vallosses).mean())
        val_rmse_epochs.append(np.array(val_rmse).mean())
        val_acc_epochs.append(np.array(val_acc).mean())

        print('Epoch:  ' + str(epoch) + '  Loss: ' + '%.3f'% np.array(losses).mean()+
         '   Val_RMSE: '+'%.3f'% np.array(val_rmse).mean()+
         '   val_acc: ' + '%.3f'% np.array(val_acc).mean()+
         '  w1, w2: '+ '%.3f' %W1 +' %.3f' %W2)

        if best_val > np.array(val_rmse).mean():
            best_val = np.array(val_rmse).mean()
            best_epoch = epoch
            torch.save(model.state_dict(), model_path +'_best'+'.pth')
            print("  valid val is improved. the model is saved.")
        else:
            if (epoch - best_epoch) >= early_stop:
                break
        if best_valmae > np.array(val_mae).mean():
            best_valmae = np.array(val_mae).mean()

    torch.save(model.state_dict(), model_path+'.pth')
    print('The best rmse: '+ '%.3f' %best_val +  '  The best mae: '+ '%.3f' %best_valmae)
    return train_rmse_epochs, train_acc_epochs, val_rmse_epochs, val_acc_epochs,  val_loss

if __name__ == '__main__':
    EPOCHES = 500
    BATCH_SIZE = 8
    channels = 4
    n_classes = 2 
    optimizer_name = 'Adam' #optimizer

    data_root =  'D:/data/mydataset/'
    model_path = 'D:/data/mydataset/model/Unetp0625'

    early_stop = 200
    torch.cuda.empty_cache()

    train_rmse_epochs, train_acc_epochs, val_rmse_epochs, val_acc_epochs,  val_loss_epochs = train(EPOCHES, BATCH_SIZE, data_root, channels, n_classes, model_path, early_stop)

    # Merge 4 list data into one DataFrame object
    data = {'train_rmse_epochs': train_rmse_epochs,'train_acc_epochs': train_acc_epochs,'val_rmse_epochs': val_rmse_epochs, 'val_acc_epochs':val_acc_epochs}
    df = pd.DataFrame(data)

    # Output DataFrame objects to Excel
    with pd.ExcelWriter('Unetplus.xlsx') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
