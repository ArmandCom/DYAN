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
from utils import videoDataset2stream
from utils import save_checkpoint
from utils import getListOfFolders
# from utils import warp
from utils import showOFs
from utils import showframes

## Import Model
from DyanOF import OFModel
from DyanOF import creatRealDictionary

############################# Import Section #################################

lam = 0.6
lamOF = 0.5

## HyperParameters for the Network
NumOfPoles = 40
EPOCH = 300
BATCH_SIZE = 1
LR = 0.001
gpu_id = 1

FRA = 6 # input number of frame
PRE = 1 # output number of frame
N_FRAME = FRA+PRE
N = NumOfPoles*4
T = FRA

#mnist
x_fra = 128
y_fra = 160


## Load saved model
# load_ckpt = True
load_ckpt = True

savecheckpt = False
saveEvery = 10
ckpt_file_OF = '/home/armandcomas/DYAN/preTrainedModel/' \
              'Kitti_ChaseOF-GL_normDic-stdx_lam2_lossFuPRE_FRA9-PRE1_Comp-ori_60.pth'
# checkptname_OF = '/home/armandcomas/DYAN/preTrainedModel/' \
#               'Kitti_ChaseOF_%s_lam%s_lossFu%s_FRA%d-PRE%d_Comp-%s_' % ('normDic-stdx', '06', 'PRE', FRA, PRE, 'ori')

ckpt_file_rgb = '/home/armandcomas/DYAN/preTrainedModel/' \
              'Kitti_ChaseRGB_normDic-stdx_lam06_lossFuPRE_FRA6-PRE1_Comp-ori_40.pth'
checkptname_rgb = '/home/armandcomas/DYAN/preTrainedModel/' \
              'Kitti_ChaseRGB_%s_lam%s_lossFu%s_FRA%d-PRE%d_Comp-%s_' % ('normDic-stdx', '06', 'PRE', FRA, PRE, 'ori')



## Load input data
# trainFolderFile = 'trainlist01.txt'
rootDir = '/home/armandcomas/datasets/Kitti_Flows/'
rootDir_rgb = '/home/armandcomas/datasets/Kitti/frames/'

# trainFoldeList = getListOfFolders(trainFolderFile)[::10]
listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]


trainingData = videoDataset2stream(folderList=listOfFolders,
                            rootDir=rootDir,
                            rootDir_rgb=rootDir_rgb,
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

def loadModel(ckpt_file):
    loadedcheckpoint = torch.load(ckpt_file)
    #model.load_state_dict(loadedcheckpoint['state_dict'])
    #optimizer.load_state_dict(loadedcheckpoint['optimizer'])
    stateDict = loadedcheckpoint['state_dict']

    # load parameters
    Dtheta = stateDict['l1.theta']
    Drr    = stateDict['l1.rr']
    model = OFModel(Drr, Dtheta, T, PRE, lam, gpu_id)
    model.cuda(gpu_id)
    Drr = Variable(Drr.cuda(gpu_id))
    Dtheta = Variable(Dtheta.cuda(gpu_id))
    # dictionary = creatRealDictionary(N_FRAME,Drr,Dtheta, gpu_id)

    return model, Drr, Dtheta

def warp(input,tensorFlow):

    torchHorizontal = torch.linspace(-1.0, 1.0, input.size(3))
    torchHorizontal = torchHorizontal.view(1, 1, 1, input.size(3)).expand(input.size(0), 1, input.size(2), input.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, input.size(2))
    torchVertical = torchVertical.view(1, 1, input.size(2), 1).expand(input.size(0), 1, input.size(2), input.size(3))

    tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda(gpu_id)
    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=input, grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

## Create the model
'''modelOF = OFModel(Drr, Dtheta, T, PRE, lamOF, gpu_id)
modelOF.cuda(gpu_id)
optimizerOF = torch.optim.Adam(modelOF.parameters(), lr=LR)
schedulerOF = lr_scheduler.MultiStepLR(optimizerOF, milestones=[50, 130], gamma=0.5)'''

ofmodel, ofDrr, ofDtheta = loadModel(ckpt_file_OF)


modelrgb = OFModel(Drr, Dtheta, T+1, PRE, lam, gpu_id)
modelrgb.cuda(gpu_id)
optimizerrgb = torch.optim.Adam(modelrgb.parameters(), lr=LR)
schedulerrgb = lr_scheduler.MultiStepLR(optimizerrgb, milestones=[50, 130], gamma=0.5)
loss_mse = nn.MSELoss()
start_epoch = 1

## If want to continue training from a checkpoint
if(load_ckpt):
    # loadedcheckpoint = torch.load(ckpt_file_OF)
    # start_epoch = loadedcheckpoint['epoch']
    # modelOF.load_state_dict(loadedcheckpoint['state_dict'])
    # optimizerOF.load_state_dict(loadedcheckpoint['optimizer'])
    loadedcheckpoint = torch.load(ckpt_file_rgb)
    start_epoch = loadedcheckpoint['epoch'] # Cuidao
    modelrgb.load_state_dict(loadedcheckpoint['state_dict'])
    optimizerrgb.load_state_dict(loadedcheckpoint['optimizer'])


print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()

count = 0
## Start the Training
for epoch in range(start_epoch, EPOCH+1):
    loss_value = []
    loss_valueR = []
    schedulerrgb.step()
    for i_batch, sample in enumerate(dataloader):

        data = sample['alframes'].squeeze(0).cuda(gpu_id)
        dataori = sample['frames'].squeeze(0).cuda(gpu_id)
        of = sample['of'].squeeze(0).cuda(gpu_id) #could be done in different GPU's
        ofori = sample['ofori'].squeeze(0)  #.cuda(gpu_id)


        expectedOut = Variable(data)
        inputData = Variable(data[:,0:N_FRAME,:])
        optimizerrgb.zero_grad()
        output = modelrgb.forward(inputData)

        with torch.no_grad():
            ofout = ofmodel.forward(Variable(of[:,0:FRA,:]))

        out = warp(output[:, FRA, :].view(3,x_fra,y_fra).unsqueeze(0), ofout[:, FRA, :].view(2,x_fra,y_fra).unsqueeze(0))
        showframes(out.view(3, x_fra, y_fra).unsqueeze(0).unsqueeze(2),
                expectedOut[:, N_FRAME, :].view(3, x_fra, y_fra).unsqueeze(0).unsqueeze(2), 1, '')

        # lossR = loss_mse(output[:, FRA, :], expectedOut[:, FRA, :])
        loss = loss_mse(output[:, FRA, :], expectedOut[:, FRA, :]) \
               + loss_mse(output[:, 0:FRA, :], expectedOut[:, 0:FRA, :])/(2*FRA) # probar amb expectedout com a GT


        loss.backward()
        optimizerrgb.step()

        loss_value.append(loss.data.item())
        # loss_valueR.append(lossR.data.item())


    loss_val = np.mean(np.array(loss_value))
    # loss_valR = np.mean(np.array(loss_valueR))

    if epoch % saveEvery ==0 and savecheckpt:
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': modelrgb.state_dict(),
                            'optimizer' : optimizerrgb.state_dict(),
                            },checkptname_rgb+str(epoch)+'.pth')

    if epoch % 30 == 0:
        print(modelrgb.state_dict()['l1.rr'])
        print(modelrgb.state_dict()['l1.theta'])
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val )#, '| train loss pred: %.4f' % loss_valR)