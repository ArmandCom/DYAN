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
from utils import save_checkpoint
from utils import getListOfFolders

## Import Model
from DyanOF import OFModel
from fast_alm import *

############################# Import Section #################################

'''FRIDAY 9: SUBTRACT PREDICTED FRAME FROM EVERY FRAME IN 0:FRA AND LOOK FOR MIN DIFF. 
LET'S SEE WHICH ONE REPLICATES'''
lam = 0.5

## HyperParameters for the Network
NumOfPoles = 40
EPOCH = 300
BATCH_SIZE = 1
LR = 0.001
gpu_id = 1  # 3?

## For training UCF
# Input -  3 Optical Flow
# Output - 1 Optical Flow
## For training Kitti
# Input -  9 Optical Flow
# Output - 1 Optical Flow

FRA = 6 # input number of frame
PRE = 1 # output number of frame
N_FRAME = FRA+PRE
N = NumOfPoles*4
T = FRA
saveEvery = 10
# N_FRAME_FOLDER = 18

#mnist
x_fra = 128
y_fra = 160

#plot options
px_ev = False # save plots of pixel evolutiona and OF inputs.


## Load saved model
load_ckpt = True
ckpt_file = 'Kitti_lam08_OF_FRA9_loss-PRE_70.pth' # for Kitti Dataset: 'KittiModel.pth'
checkptname = "test"
# checkptname = "Kitti_lam05_OF_FRA9_loss-all_"



## Load input data
# trainFolderFile = 'trainlist01.txt'
rootDir = '/home/armandcomas/datasets/Kitti_Flows/'

# trainFoldeList = getListOfFolders(trainFolderFile)[::10]
listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]


trainingData = videoDataset(folderList=listOfFolders,
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
# What and where is gamma

## Create the model
model = OFModel(Drr, Dtheta, T, PRE, lam, gpu_id)
model.cuda(gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.5) # if Kitti: milestones=[100,150]
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
        expectedOut = Variable(data)
        inputData = Variable(data[:,0:FRA,:])
        optimizer.zero_grad()
        output = model.forward(inputData)
        loss = loss_mse(output[:, FRA], expectedOut[:, FRA]) # if Kitti: loss = loss_mse(output, expectedOut)
        lossR = loss_mse(output[:, 0:FRA], expectedOut[:, 0:FRA])

        loss.backward()
        optimizer.step()

        loss_value.append(loss.data.item())
        loss_valueR.append(lossR.data.item())


        ''' ADMM + PLOT '''
        Lambda = 100
        px = 10300
        sequence = expectedOut[:,:,px]
        Omega = torch.Tensor(np.array([1, 1, 1, 1, 1, 1, 0]))
        estimate = l2_fast_alm(sequence, Lambda, Omega)
        estimateDyan = output[:,:,px]
        plt.figure()
        t = np.arange(0, FRA+PRE, 1)
        gt, = plt.plot(t, sequence[0,:].data.cpu().detach().numpy(), label="GT")
        d0, = plt.plot(t, estimateDyan[0,:].data.cpu().detach().numpy(), '-.r*', label="Dyan")
        d1, = plt.plot(t, estimate[0,:].data.cpu().detach().numpy(), ':b', label="Hankel")
        # d2, = plt.plot(t, coefP2[idxcp], '--m', label="lam 0.1, standardized data")
        # plt.ylim((7, 8.2))
        plt.legend(handles=[gt, d1, d0])
        plt.savefig('Ideal_estimate.png')
        plt.close()

        '''PLOT FRAMES'''
        po = output.data.cpu().numpy()
        eo = expectedOut.data.cpu().numpy()

        tmp1 = np.zeros([128, 160, 3], dtype=np.float16)
        tmp1[:, :, 0] = po[0, FRA, :].reshape(x_fra, y_fra)
        tmp1[:, :, 1] = po[1, FRA, :].reshape(x_fra, y_fra)
        scipy.misc.imsave('predicted_outputOF.png', tmp1)

        tmp2 = np.zeros([128, 160, 3], dtype=np.float16)
        tmp2[:, :, 0] = eo[0, FRA, :].reshape(x_fra, y_fra)
        tmp2[:, :, 1] = eo[1, FRA, :].reshape(x_fra, y_fra)
        scipy.misc.imsave('expected_outputOF.png', tmp2)

    loss_val = np.mean(np.array(loss_value))
    loss_valR = np.mean(np.array(loss_valueR))

    if epoch % saveEvery ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')

    if epoch % 30 == 0:
        print(model.state_dict()['l1.rr'])
        print(model.state_dict()['l1.theta'])
        # loss_val = float(loss_val/i_batch)
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| train loss rec: %.4f' % loss_valR)