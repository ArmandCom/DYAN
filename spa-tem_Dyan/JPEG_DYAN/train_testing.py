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

## Dependencies classes and functions
from utils import gridRing
from utils import asMinutes
from utils import timeSince
from utils import getWeights
from utils import videoDataset
from utils import save_checkpoint
from utils import getListOfFolders
from utils import idct2

## Import Model
from DyanOF import OFModel
from DyanOF_1 import OFModel1
from DyanOF_2 import OFModel2

import matplotlib.pyplot as plt
import scipy

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc # pip install Pillow
import matplotlib.pylab as pylab


############################# Import Section #################################

blockSize = 8
lam = 0.1
lambd1 = 0.1
lambd2 = 0.1

## HyperParameters for the Network
NumOfPoles = 40
CLen = NumOfPoles*4 + 1
EPOCH = 300
BATCH_SIZE = 1
LR = 0.0015
gpu_id = 0


FRA = 5 # input number of frame
PRE = 1 # output number of frame
N_FRAME = FRA+PRE
N = NumOfPoles*4
T = FRA
saveEvery = 5

#mnist
x_fra = 480
y_fra = 640


## Load saved model 
load_ckpt = False
ckpt_file = '/home/armandcomas/DYAN/preTrainedModel/Kitti_DCT_lam005_Frames_FRA7_Caltech_loss-NFrames-last4_10.pth' # for Kitti Dataset: 'KittiModel.pth'
# checkptname = "UCFModel"
rootCkpt = '/home/armandcomas/DYAN/preTrainedModel/'

checkpt = 'Kitti_DCT_lam01_Frames-initial_FRA5_Caltech_loss-weighted_inputin01'
checkptname = os.path.join(rootCkpt, checkpt)

checkpt1 = 'Kitti_DCT_lam01_Frames-mean_FRA5_Caltech_loss-weighted_inputin01'
checkptname1 = os.path.join(rootCkpt, checkpt1)

checkpt2 = 'Kitti_DCT_lam01_Frames-meanstd_FRA5_Caltech_loss-weighted_inputin01'
checkptname2 = os.path.join(rootCkpt, checkpt2)

## Load input data

# rootDir = '/home/armandcomas/datasets/Kitti_Flows/'
rootDir = '/home/armandcomas/datasets/Caltech/images/'
listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]
trainingData = videoDataset(folderList=listOfFolders,
                            rootDir=rootDir,
                            blockSize=blockSize,
                            nfra = N_FRAME)
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

## Create the time model
model = OFModel(Drr, Dtheta, T, PRE, lam, gpu_id)
model.cuda(gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

model1 = OFModel1(Drr, Dtheta, T, PRE, lambd1, gpu_id)
model1.cuda(gpu_id)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=LR)
scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[100,150], gamma=0.1)

model2 = OFModel2(Drr, Dtheta, T, PRE, lambd2, gpu_id)
model2.cuda(gpu_id)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR)
scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[100,150], gamma=0.1)

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

    t0_epoch = time.time()
    loss_value  = []
    loss_value1 = []
    loss_value2 = []
    # loss_img_value = []
    scheduler.step()

    for i_batch, sample in enumerate(dataloader):

        data = sample['frames'].squeeze(0).cuda(gpu_id)
        flows = sample['flows'].squeeze(0)

        expectedOut = Variable(data)
        inputData = Variable(data[:,0:FRA,:])
        optimizer.zero_grad()

        y = data[:, 0:FRA+PRE, :].data

        dct = model.forward(inputData)
        dct1 = model1.forward(inputData)
        dct2 = model2.forward(inputData)

        '''Plot DCT coefficients'''
        coef = dct.data
        coef1 = dct1.data
        coef2 = dct2.data
        coefP = torch.t(coef[0,:,:]).data.cpu().detach().numpy()
        coefP1 = torch.t(coef1[0, :, :]).data.cpu().detach().numpy()
        coefP2 = torch.t(coef2[0, :, :]).data.cpu().detach().numpy()
        coefR = torch.t(y[0,:,:])

        maxcp, idxcp = coefR[:,0].max(0)
        plt.figure()
        t = np.arange(0, FRA+PRE, 1)
        gt, = plt.plot(t, coefR[idxcp].data.cpu().detach().numpy(), label="GT")
        d0, = plt.plot(t, coefP[idxcp], '-.r*', label="0")
        d1, = plt.plot(t, coefP1[idxcp], ':b', label="1")
        # d2, = plt.plot(t, coefP2[idxcp], '--m', label="lam 0.1, standardized data")
        # plt.ylim((7, 8.2))
        plt.legend(handles=[gt, d1, d0])
        plt.savefig('max_dct_coef_values.png')
        plt.close()

        '''LOSSES'''
        loss = loss_mse(dct[:, 0:FRA, :], expectedOut[:, 0:FRA, :])/FRA + loss_mse(dct[:, FRA, :], expectedOut[:, FRA, :])
        loss.backward()
        optimizer.step()
        loss_value.append(loss.data.item())

        loss1 = loss_mse(dct1[:, 0:FRA, :], expectedOut[:, 0:FRA, :])/FRA + loss_mse(dct1[:, FRA, :], expectedOut[:, FRA, :])
        loss1.backward()
        optimizer1.step()
        loss_value1.append(loss1.data.item())

        loss2 = loss_mse(dct2[:, 0:FRA, :], expectedOut[:, 0:FRA, :])/FRA + loss_mse(dct2[:, FRA, :], expectedOut[:, FRA, :])
        loss2.backward()
        optimizer2.step()
        loss_value2.append(loss2.data.item())

        '''RECONSTRUCT IMAGE'''
        '''# dct_img = dct[:,FRA,:]
        dct_img = dct[:, FRA, :]
        dct_img = dct_img.cpu().detach().numpy().reshape(-1,x_fra,y_fra)
        dct_img = np.transpose(dct_img, (1,2,0))
        imgsize = dct_img.shape
        img_idct = np.zeros(imgsize)  # .astype(float)

        for i in r_[:imgsize[0]:blockSize]:
            for j in r_[:imgsize[1]:blockSize]:
                img_idct[i:(i + blockSize), j:(j + blockSize)] = idct2(dct_img[i:(i + blockSize), j:(j + blockSize)])
                # img_idct[i:(i + blockSize), j:(j + blockSize)] = np.fft.ifft2(dct_img[i:(i + blockSize), j:(j + blockSize)])
    
        img_idct_t = torch.from_numpy(img_idct)
        loss_img = loss_mse(img_idct_t, flows[FRA,:,:,:].permute(1,2,0))
        loss_img_value.append(loss_img)'''
        '''PLOT for flows'''
        '''# po = img_idct
        # eo = flows[FRA, :, :, :].data.cpu().numpy()
        # peo = flows[FRA-1, :, :, :].data.cpu().numpy()
        # # #
        #
        # tmp1 = np.zeros([128, 160, 3], dtype=np.float16)
        # tmp1[:, :, 0] = po[:, :, 0]
        # tmp1[:, :, 1] = po[:, :, 1]
        # scipy.misc.imsave('predicted_outputOF.png', tmp1)
        # #
        # tmp2 = np.zeros([128, 160, 3], dtype=np.float16)
        # tmp2[:, :, 0] = eo[0, :, :]
        # tmp2[:, :, 1] = eo[1, :, :]
        # scipy.misc.imsave('expected_outputOF.png', tmp2)
        #
        # tmp3 = np.zeros([128, 160, 3], dtype=np.float16)
        # tmp3[:, :, 0] = peo[0, :, :]
        # tmp3[:, :, 1] = peo[1, :, :]
        # scipy.misc.imsave('previous_expected_outputOF.png', tmp3)
        #
        # # tmp = np.zeros([3, 128, 160, 3], dtype=np.float16)
        # tmp = np.concatenate((tmp3, tmp2, tmp1), axis=0)
        # scipy.misc.imsave('outputs.png', tmp)

        PLOT for frames
        po = img_idct
        eo = flows[FRA, :, :, :].data.cpu().numpy()
        peo = flows[FRA-1, :, :, :].data.cpu().numpy()
        # #

        tmp1 = np.zeros([x_fra, y_fra, 3], dtype=np.float16)
        tmp1 = po
        # scipy.misc.imsave('predicted_output_dct.png', tmp1)
        
        tmp2 = np.zeros([x_fra, y_fra, 3], dtype=np.float16)
        tmp2[:, :, 0] = eo[0, :, :]
        tmp2[:, :, 1] = eo[1, :, :]
        tmp2[:, :, 2] = eo[2, :, :]
        # scipy.misc.imsave('expected_output_dct.png', tmp2)

        tmp3 = np.zeros([x_fra, y_fra, 3], dtype=np.float16)
        tmp3[:, :, 0] = peo[0, :, :]
        tmp3[:, :, 1] = peo[1, :, :]
        tmp3[:, :, 2] = peo[2, :, :]
        # scipy.misc.imsave('previous_expected_output_dct.png', tmp3)

        tmp = np.concatenate((tmp3, tmp2, tmp1), axis=0)
        scipy.misc.imsave('outputsDCT.png', tmp)'''

    loss_val = np.mean(np.array(loss_value))
    loss_val1 = np.mean(np.array(loss_value1))
    loss_val2 = np.mean(np.array(loss_value2))



    ## Checkpoint + prints
    # if epoch % saveEvery ==0 :
    #     save_checkpoint({	'epoch': epoch + 1,
    #                         'state_dict': model.state_dict(),
    #                         'optimizer' : optimizer.state_dict(),
    #                         },checkptname+str(epoch)+'.pth')
    #     save_checkpoint({	'epoch': epoch + 1,
    #                         'state_dict': model1.state_dict(),
    #                         'optimizer' : optimizer1.state_dict(),
    #                         },checkptname1+str(epoch)+'.pth')
    #     save_checkpoint({	'epoch': epoch + 1,
    #                         'state_dict': model2.state_dict(),
    #                         'optimizer' : optimizer2.state_dict(),
    #                         },checkptname2+str(epoch)+'.pth')
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| train loss 1: %.4f' % loss_val1, '| train loss 2: %.4f' % loss_val2,'| time per epoch: %.4f' % (time.time()-t0_epoch), '| Checkpoint name: %s' % checkpt)