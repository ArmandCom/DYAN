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
import scipy

## Dependencies classes and functions
from utils import gridRing
from utils import asMinutes
from utils import timeSince
from utils import getWeights
from utils import videoDataset
from utils import save_checkpoint
from utils import getListOfFolders


## Import Model
from DyanOFST_SW4ch import OFModelST
from DyanOF import OFModel


############################# Import Section #################################

devices = [0, 3]
dev_idx = 3
# dev_idx_T = 0

## HyperParameters for the Network
NumOfPoles = 40
CLen = NumOfPoles*4 + 1
EPOCH = 150
BATCH_SIZE = 1
LR = 0.0015
gpu_id = 1
gpu_id2 = 3 # parallelize computation

FRA = 9 # input number of frame
PRE = 1 # output number of frame
N_FRAME = FRA+PRE
N = NumOfPoles*4
T = FRA# Length of spacial window
saveEvery = 5

#mnist
x_fra = 128
y_fra = 160


## Load saved model
load_ckpt = False
ckpt_file = '' # for Kitti Dataset: 'KittiModel.pth'
# checkptname = "UCFModel"
checkptname = "/home/armandcomas/DYAN/preTrainedModel/Kitti_4ch-ST_lam0.1_nChu8_Sep-losses_"
# checkptname = "/home/armandcomas/DYAN/preTrainedModel/Kitti_Normal-ST-test_2_"



## Load input data
rootDir = '/home/armandcomas/datasets/Kitti_Flows/'
listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]
trainingData = videoDataset(folderList=listOfFolders, rootDir=rootDir)
dataloader = DataLoader(trainingData,
                        batch_size=BATCH_SIZE ,
                        shuffle=True, num_workers=1)

## Initializing r, theta
P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

# Create the model
model = OFModelST(Drr, Dtheta, Drr, Dtheta, Drr, Dtheta, T, PRE, x_fra, y_fra)
model = nn.DataParallel(model, device_ids=devices, output_device=dev_idx).cuda()
# model.cuda(dev_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# modelT = OFModel(Drr, Dtheta, T, PRE, dev_idx_T)
# modelT.cuda(dev_idx_T)
# optimizerT = torch.optim.Adam(modelT.parameters(), lr=LR)
# schedulerT = lr_scheduler.MultiStepLR(optimizerT, milestones=[100, 150], gamma=0.1)

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

for epoch in range(start_epoch, EPOCH+1):
    t0_epoch = time.time()

    loss_value = []
    scheduler.step()

    for i_batch, sample in enumerate(dataloader):

        data = sample['frames'].squeeze(0).cuda(dev_idx)
        expectedOut = Variable(data)
        optimizer.zero_grad()

        # 4 channel Spacial model
        # outH, outV, outHI, outVI = model(data)
        output = model(data)
        ## REGULAR MODEL
        # dataT = data.cuda(dev_idx_T)
        # optimizerT.zero_grad()
        # out = modelT.forward(Variable(dataT[:,0:FRA,:]))
        # lossT = loss_mse(out[:, FRA], expectedOut[:, FRA])
        # lossT.backward()
        # optimizerT.step()

        '''IMPLEMENTATION OF LOSS AND BACKPROP SEQUENTIAL'''
        loss1 = loss_mse(outH,  expectedOut[:,FRA,:].view(-1, x_fra, y_fra))
        loss2 = loss_mse(outV,  expectedOut[:,FRA,:].view(-1, x_fra, y_fra))
        loss3 = loss_mse(outHI, expectedOut[:,FRA,:].view(-1, x_fra, y_fra))
        loss4 = loss_mse(outVI, expectedOut[:,FRA,:].view(-1, x_fra, y_fra))

        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward(retain_graph=True)
        loss4.backward()
        optimizer.step()
        loss = loss_mse((outH + outV + outHI + outVI)/4, expectedOut[:,FRA,:].view(-1, x_fra, y_fra))
        # + out[:, FRA].view(-1, x_fra, y_fra).cuda(dev_idx)
        loss_value.append(loss.data.item())

        '''Save images to see if it's actually working'''
        po = outV.data.cpu().numpy()
        eo = expectedOut.data.cpu().numpy()
        #
        tmp1 = np.zeros([128, 160, 3], dtype=np.float16)
        tmp1[:, :, 0] = po[0, :, :]
        tmp1[:, :, 1] = po[1, :, :]
        scipy.misc.imsave('predicted_outputOF.png', tmp1)
        #
        tmp2 = np.zeros([128, 160, 3], dtype=np.float16)
        tmp2[:, :, 0] = eo[0, FRA, :].reshape(x_fra, y_fra)
        tmp2[:, :, 1] = eo[1, FRA, :].reshape(x_fra, y_fra)
        scipy.misc.imsave('expected_outputOF.png', tmp2)

    loss_val = np.mean(np.array(loss_value))

    ## Checkpoint + prints
    if epoch % saveEvery ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| time per epoch: %.4f' % (time.time()-t0_epoch))














##  PRED DIM0 IS DUPLICATED (device 1 and 0) - WHEN CHANGING NUMBER OF GPUS IS STILL DUPLICATED
''' LOSS GOES DOWN VERY SLOW AND TIME IS SUPERIOR THAN WITHOUT PARALLELIZATION -  THIS COULD BE DUE TO SMALL BATCH
    DATAPARALLEL DOESN'T USE ALL PROVIDED DEVICES'''
'''IMPLEMENTATION OF LOSS AND BACKPROP PARALLEL 1'''
# for i in range(len(devices)):
#     loss = loss_mse(pred[i,:,:], expectedOut[:,FRA])
#     loss.backward(retain_graph=True)
#     if(i == len(devices)-1):
#         loss.backward()
#     optimizer.step()
#     loss_value.append(loss.data.item())
'''IMPLEMENTATION OF LOSS AND BACKPROP PARALLEL 2'''
# expOut = torch.empty(len(devices),2,20480).cuda(devices[0])
# for i in range(len(devices)):
#     expOut[i,:,:] = expectedOut[:,FRA]
# loss = loss_mse(pred, expectedOut)
# loss.backward()
# optimizer.step()
# loss_value.append(loss.data.item())
'''IMPLEMENTATION OF LOSS AND BACKPROP PARALLEL 3'''  # Add loss inside model (backprop outside I guess)