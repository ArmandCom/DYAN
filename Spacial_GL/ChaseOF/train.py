############################# Import Section #################################
import sys
## Imports related to PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import scipy.misc

import matplotlib
matplotlib.use('Agg')
import random

import matplotlib.pyplot as plt

## Generic imports
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

## Dependencies classes and functions
from utils import gridRing
from utils import asMinutes
from utils import timeSince
from utils import getWeights
from utils import videoDataset
from utils import save_checkpoint
from utils import getListOfFolders
# from utils import warp
from utils import showOFs

## Import Model
from DyanOF import OFModel

############################# Import Section #################################

lam = 0.6

## HyperParameters for the Network
NumOfPoles = 40
EPOCH = 300
BATCH_SIZE = 1
LR = 0.001
gpu_id = 1

FRA = 3 # input number of frame
PRE = 1 # output number of frame
N_FRAME = FRA+PRE
N = NumOfPoles*4
T = FRA

# N_FRAME_FOLDER = 18

#mnist
x_fra = 240
y_fra = 320

#plot options
px_ev = False


## Load saved model
# load_ckpt = True
load_ckpt = False
savecheckpt = True
saveEvery = 10
ckpt_file = '/home/armandcomas/DYAN/preTrainedModel/' \
              'Kitti_ChaseOF-GL_normDic-stdx_lam2_lossFuPRE_FRA9-PRE1_Comp-ori_60.pth'
checkptname = '/home/armandcomas/DYAN/preTrainedModel/' \
              '%s_ChaseOF_%s_lam%s_lossFu%s_FRA%d-PRE%d' % ('UCF', 'std-input', '06', 'PRE', FRA, PRE)
# checkptname = "Kitti_lam05_OF_FRA9_loss-all_"



## Load input data
# trainFolderFile = 'trainlist01.txt'

# rootDir = '/home/armandcomas/datasets/Kitti_Flows/'
rootDir = '/home/armandcomas/datasets/' \
          'UCF_Flows' # UCF

## FOR UCF
trainFolderFile = './trainlist01.txt'
trainFoldeList = getListOfFolders(trainFolderFile)[::10]

## FOR KITTI
# listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]

trainingData = videoDataset(folderList=trainFoldeList,
                            rootDir=rootDir,
                            N_FRAME = N_FRAME)

dataloader = DataLoader(trainingData,
                        batch_size=BATCH_SIZE ,
                        shuffle=True, num_workers=1)

## Initializing r, theta
P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

# def warp(input,tensorFlow):
#
#     torchHorizontal = torch.linspace(-1.0, 1.0, input.size(3))
#     torchHorizontal = torchHorizontal.view(1, 1, 1, input.size(3)).expand(input.size(0), 1, input.size(2), input.size(3))
#     torchVertical = torch.linspace(-1.0, 1.0, input.size(2))
#     torchVertical = torchVertical.view(1, 1, input.size(2), 1).expand(input.size(0), 1, input.size(2), input.size(3))
#
#     tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda(gpu_id)
#     tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0) ], 1)
#
#     return torch.nn.functional.grid_sample(input=input, grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

## Create the model
model = OFModel(Drr, Dtheta, T, PRE, lam, gpu_id)
model.cuda(gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.5) # if Kitti: milestones=[100,150]
loss_mse = nn.MSELoss()
start_epoch = 1

## If want to continue training from a checkpoint
if(load_ckpt):
    loadedcheckpoint = torch.load(ckpt_file)
    start_epoch = loadedcheckpoint['epoch']
    model.load_state_dict(loadedcheckpoint['state_dict'])
    optimizer.load_state_dict(loadedcheckpoint['optimizer'])


print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()

count = 0
## Start the Training
for epoch in range(start_epoch, EPOCH+1):
    loss_value = []
    loss_valueR = []
    scheduler.step()
    for i_batch, sample in enumerate(dataloader):

        data = sample['frames'].squeeze(0).cuda(gpu_id)
        dataori = sample['framesori'].squeeze(0).cuda(gpu_id)
        expectedOut = Variable(data)
        inputData = Variable(data[:,0:FRA,:])
        optimizer.zero_grad()
        output = model.forward(inputData)

        '''Per PRE > 1: 
        1- Mirar que nomes alinii els FRA inputs.
        2- Quan predigui els seguents PRE, recalc FISTA (?) --> Si
        2b- Quan predigui els seguents PRE, warp amb les pred anteriors + actual --> + noisy
        Aixo es fa en test!'''
        if random.randint(1, 100) == 1:
            showOFs(output[:, FRA, :].view(2, x_fra, y_fra).unsqueeze(0).unsqueeze(2),
                    expectedOut[:, FRA, :].view(2, x_fra, y_fra).unsqueeze(0).unsqueeze(2), 1, '')
        # showOFs(output[:, FRA, :].view(2, x_fra, y_fra).unsqueeze(0).unsqueeze(2),
        #         dataori[:, FRA-1, :].view(2, x_fra, y_fra).unsqueeze(0).unsqueeze(2), 1)

        # pred = warp(expectedOut[:, FRA-1].view(2,x_fra,y_fra).unsqueeze(0), output[:, FRA].view(2,x_fra,y_fra).unsqueeze(0))
        # lossR = loss_mse(output[:, 0:FRA, :], expectedOut[:, 0:FRA, :])
        loss = loss_mse(output[:, FRA, :], expectedOut[:, FRA, :])
               #+ loss_mse(output[:, 0:FRA, :], expectedOut[:, 0:FRA, :])/(2*FRA)


        loss.backward()
        optimizer.step()

        loss_value.append(loss.data.item())
        # loss_valueR.append(lossR.data.item())


    loss_val = np.mean(np.array(loss_value))
    # loss_valR = np.mean(np.array(loss_valueR))

    if epoch % saveEvery ==0 and savecheckpt:
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')

    if epoch % 30 == 0:
        print(model.state_dict()['l1.rr'])
        print(model.state_dict()['l1.theta'])
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val )#, '| train loss pred: %.4f' % loss_valR)