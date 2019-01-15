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
# from DyanOF import OFModel
# from DyanOF import Encoder
# from DyanOF import Decoder
from DyanOFST_SW import OFModel


############################# Import Section #################################



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
checkptname = "/home/armandcomas/DYAN/preTrainedModel/Kitti_2ch-ST_lam0.05_nChu4_Sep-losses_"
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

# ## Create the time model
# model_ti = OFModel(Drr, Dtheta, T, PRE, gpu_id)
# model_ti.cuda(gpu_id)
# optimizer = torch.optim.Adam(model_ti.parameters(), lr=LR)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1) # if Kitti: milestones=[100,150]
# loss_mse = nn.MSELoss()
# start_epoch = 1
# ## Create the spatial model X
# Encoder_spX = Encoder(Drr, Dtheta, y_fra, gpu_id) # change by full model?
# Encoder_spX.cuda(gpu_id) #parallelize it?
# optimizer_spX = torch.optim.Adam(Encoder_spX.parameters(), lr=LR)
# scheduler_spX = lr_scheduler.MultiStepLR(optimizer_spX, milestones=[50,100], gamma=0.1) # Parameters?
# ## Create the spatial model Y
# Encoder_spY = Encoder(Drr, Dtheta, x_fra, gpu_id)
# Encoder_spY.cuda(gpu_id) #parallelize it?
# optimizer_spY = torch.optim.Adam(Encoder_spY.parameters(), lr=LR)
# scheduler_spY = lr_scheduler.MultiStepLR(optimizer_spY, milestones=[50,100], gamma=0.1) # Parameters?


# Create the model

devices = [0, 1, 2, 3]
dev_idx = 3
model = OFModel(Drr, Dtheta, Drr, Dtheta, Drr, Dtheta, T, PRE, x_fra, y_fra)
# model = nn.DataParallel(model, device_ids=devices)
model.cuda(devices[dev_idx])
# model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1) # if Kitti: milestones=[100,150]
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

        data = sample['frames'].squeeze(0).cuda(devices[dev_idx])
        expectedOut = Variable(data)
        optimizer.zero_grad()

        ##__________Encode_Spatial__________
        # out = torch.empty(2, FRA, (x_fra+y_fra)*CLen).cuda(gpu_id)
        #
        # for i in range(FRA):
        #     t0_sp = time.time()
        #
        #     inputFrame = Variable(data[:,i,:].view(-1,x_fra,y_fra))
        #
        #
        #     outputH,outputV = model.module.forwardE(inputFrame.permute(0,2,1),inputFrame)
        #     output = torch.cat((outputH.view(-1, CLen * x_fra), outputV.view(-1, CLen * y_fra)), 1)
        #     out[:, i, :] = output
        #
        #     # print('Time for spacial encoding in time T: ', time.time() - t0_sp)
        #
        # ##____________Temporal____________
        # cPred = model.module.forward(out)
        #
        # cPredH = cPred[:, FRA, 0:(CLen * x_fra)].view(2, CLen, x_fra)
        # cPredV = cPred[:, FRA, (CLen * x_fra): ].view(2, CLen, y_fra)
        #
        # ##__________Decode_Spatial__________
        #
        # outH, outV = model.module.forwardD(cPredH, cPredV)
        # pred = ((outH.permute(0, 2, 1) + outV) / 2).view(-1, x_fra * y_fra)
        #
        # # print('Time per forward: ', time.time() - t0_epoch)

        ##All from same model
        outH, outV, outH_rec, outV_rec = model(data)

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

        '''IMPLEMENTATION OF LOSS AND BACKPROP PARALLEL 3'''
        # Add loss inside model (backprop outside I guess)

        '''IMPLEMENTATION OF LOSS AND BACKPROP SEQUENTIAL'''
        loss1 = loss_mse(outH, expectedOut[:,FRA,:].view(-1, x_fra, y_fra))
        loss2 = loss_mse(outV, expectedOut[:,FRA,:].view(-1, x_fra, y_fra))
        loss1.backward(retain_graph=True)
        loss2.backward()
        optimizer.step()
        loss = loss_mse((outH + outV)/2, expectedOut[:,FRA,:].view(-1, x_fra, y_fra))
        loss_value.append(loss.data.item())

        '''Save images to see if it's actually working'''
        po = outH_rec.data.cpu().numpy()
        eo = expectedOut.data.cpu().numpy()
        #
        tmp1 = np.zeros([128, 160, 3], dtype=np.float16)
        tmp1[:, :, 0] = po[0, :, :]
        tmp1[:, :, 1] = po[1, :, :]
        scipy.misc.imsave('predicted_outputOF.png', tmp1)
        #
        tmp2 = np.zeros([128, 160, 3], dtype=np.float16)
        tmp2[:, :, 0] = eo[0, FRA-1, :].reshape(x_fra, y_fra)
        tmp2[:, :, 1] = eo[1, FRA-1, :].reshape(x_fra, y_fra)
        scipy.misc.imsave('expected_outputOF.png', tmp2)

    loss_val = np.mean(np.array(loss_value))

    ## Checkpoint + prints
    if epoch % saveEvery ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| time per epoch: %.4f' % (time.time()-t0_epoch))