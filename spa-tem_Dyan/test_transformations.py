import os
import sys
import torch
import torch.nn as nn
import numpy as np
# TODO test sliding window --> it's necessary to avoid transitions

x = torch.arange(4*12).float().view(2,4,6)

'''TEST REVERSE'''
# reverse x
# x = x[:, np.linspace(4, 1, num=4)-1, :]

'''TEST INITIAL CONDITION SUBSTRACTION'''
# x_unbiased = x - x[:,0,:].unsqueeze(1)

'''TEST MEAN and STDEV SUBSTRACTION'''
# x_unbiased = x - torch.mean(x,1).unsqueeze(1)
# x_unbiased = x / (x.std(1).unsqueeze(1) + torch.arange(12).float().view(2,1,6))
# x_unbiased = x.std(1).unsqueeze(1)+ torch.arange(12).float().view(2,1,6)


'''TEST MAX'''
# max, locmax = x[0,:,0].max(0)

# reverse y
# x = x.permute(0,2,1)[:, np.linspace(6, 1, num=6)-1, :]

'''TEST UNFOLD'''
# xufh = x.unfold(1, 2, 2)
# xufhc = torch.cat(list(xufh), dim=0).permute(0,2,1)
# If for each channel we get 4 chunks it will
# concatenate the blocks of 4 chunks for each channel
# 1234,1234

# xfh = xufhc.contiguous().view(2,4,6)
# xfh = xufhc.contiguous().view(2,6,4)

# From test reverse
# yfh = xfh[:, np.linspace(4, 1, num=4)-1, :]
# yfh = xfh[:, np.linspace(6, 1, num=6)-1, :].permute(0,2,1)

'''TEST UNFOLD BLOCKS'''
xufhv = x.unfold(2, 2, 2).unfold(1,2,2).permute(1,0,2,3,4)
xufhc = torch.cat(list(xufhv), dim=1)
# If for each channel we get 4 chunks it will
# concatenate the blocks of 4 chunks for each channel
# 1234,1234

# xfh = xufhc.contiguous().view(2,2,6,2).contiguous().view(2,6,4)
xfh = xufhc.view(2,2,6,2).permute(0,1,3,2).contiguous().view(2,4,6)

'''TEST UNFOLD SLIDING WINDOW'''
# xufh = x.unfold(1, 3, 1)
# xufhc = torch.cat(list(xufh), dim=0).permute(0,2,1)
# tmp1 = xufhc[0,:,:].unsqueeze(0)
# tmp2 = xufhc[2,:,:].unsqueeze(0)
# xufhc1 = torch.cat([tmp1, xufhc[1:2,2,:].unsqueeze(1)], dim=1)
# xufhc2 = torch.cat([tmp2, xufhc[3:4,2,:].unsqueeze(1)], dim=1)
# xufh = torch.cat([xufhc1, xufhc2], dim=0)

'''PRINTS'''
print(x)
# print(x_unbiased)
# print(x_unbiased.shape)
print(xufhc, xufhc.shape)
print(xufhv, xufhv.shape)
print(xfh)
# print(yfh)

# print(xufhc)
