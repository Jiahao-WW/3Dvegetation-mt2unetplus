import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU() #
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offsetY = outputs2.size()[2] - inputs1.size()[2]
        offsetX = outputs2.size()[3] - inputs1.size()[3]
        # padding = 2 * [offset // 2, offset // 2]
        # outputs1 = F.pad(inputs1, padding)
        outputs1 = nn.functional.pad(inputs1, [offsetX // 2, offsetX - offsetX//2,
                                    offsetY // 2, offsetY - offsetY//2])
        return self.conv(torch.cat([outputs1, outputs2], 1))
    
    
class unetUpC(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUpC, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size*2, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offsetY = outputs2.size()[2] - inputs1.size()[2]
        offsetX = outputs2.size()[3] - inputs1.size()[3]
        outputs1 = nn.functional.pad(inputs1, [offsetX // 2, offsetX - offsetX//2,
                                    offsetY // 2, offsetY - offsetY//2])
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetUpsimple(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUpsimple, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        return self.conv(self.up(x))


