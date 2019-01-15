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

## HyperParameters for the Network
NumOfPoles = 40
CLen = NumOfPoles*4 + 1
EPOCH = 300
BATCH_SIZE = 1
LR = 0.0015
gpu_id = 3


FRA = 7 # input number of frame
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
ckpt_file = '/home/armandcomas/DYAN/preTrainedModel/Kitti_DCT_lam2_Frames_FRA7_Caltech_loss-NFrames_10.pth' # for Kitti Dataset: 'KittiModel.pth'
# checkptname = "UCFModel"
rootCkpt = '/home/armandcomas/DYAN/preTrainedModel/'
checkpt = 'Kitti_DCT_lam005_Frames_FRA7_Caltech_loss-NFrames-last4_'
checkptname = os.path.join(rootCkpt, checkpt)



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
model = OFModel(Drr, Dtheta, T, PRE, gpu_id)
model.cuda(gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
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
    loss_value = []
    loss_img_value = []
    scheduler.step()

    for i_batch, sample in enumerate(dataloader):

        data = sample['frames'].squeeze(0).cuda(gpu_id)
        flows = sample['flows'].squeeze(0)

        expectedOut = Variable(data)
        inputData = Variable(data[:,0:FRA,:])
        optimizer.zero_grad()

        y = data[:, 0:FRA+PRE, :].data

        dct = model.forward(inputData)

        coef = dct.data
        coefP = torch.t(coef[0,:,:])
        coefR = torch.t(y[0,:,:])

        maxcp, idxcp = coefR[:,0].max(0)
        plt.figure()
        t = np.arange(0, FRA+PRE, 1)
        plt.plot(t,coefP[idxcp].data.cpu().detach().numpy(),t,coefR[idxcp].data.cpu().detach().numpy())
        plt.plot(t, coefP[idxcp+1].data.cpu().detach().numpy(), t, coefR[idxcp+1].data.cpu().detach().numpy())
        plt.savefig('max_dct_coef_values.png', dpi=200)
        plt.close()

        plt.figure()
        plt.pcolormesh(torch.cat((coefR,coefP), dim=1), cmap='RdBu')
        plt.colorbar()
        # plt.savefig('dct_evolution_Real_Pred.png', dpi=200)
        plt.show()
        plt.close()

        loss = loss_mse(dct[:, 4:N_FRAME, :], expectedOut[:, 4:N_FRAME, :])
        loss.backward()
        optimizer.step()
        loss_value.append(loss.data.item())

        # dct_img = dct[:,FRA,:]
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
        loss_img_value.append(loss_img)

        '''Print for flows'''
        # po = img_idct
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

        '''Print for frames'''
        po = img_idct
        eo = flows[FRA, :, :, :].data.cpu().numpy()
        peo = flows[0, :, :, :].data.cpu().numpy()
        # #

        tmp1 = np.zeros([x_fra, y_fra, 3], dtype=np.float16)
        tmp1 = po
        # scipy.misc.imsave('predicted_output_dct.png', tmp1)
        #
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
        scipy.misc.imsave('outputsDCT.png', tmp)

        '''SHOW RESULTS OF FRA, FRA-1, PRED: UNWARPED AND OF (BEFORE AND AFTER RECONSTRUCTION)'''
    loss_img_val = np.mean(np.array(loss_img_value))
    # print(loss_img_val)
    loss_val = np.mean(np.array(loss_value))


    ## Checkpoint + prints
    if epoch % saveEvery ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| time per epoch: %.4f' % (time.time()-t0_epoch), '| Checkpoint name: %s' % checkpt)