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

############################# Import Section #################################



## HyperParameters for the Network
NumOfPoles = 40
EPOCH = 300
BATCH_SIZE = 1
LR = 0.0015
gpu_id = 1  # 3?

## For training UCF
# Input -  3 Optical Flow
# Output - 1 Optical Flow
## For training Kitti
# Input -  9 Optical Flow
# Output - 1 Optical Flow

FRA = 3 # input number of frame
PRE = 1 # output number of frame
N_FRAME = FRA+PRE
N = NumOfPoles*4
T = FRA # number of row in dictionary(same as input number of frame)
saveEvery = 2

#mnist
x_fra = 64
y_fra = 64

#plot options
px_ev = False # save plots of pixel evolutiona and OF inputs.


## Load saved model 
load_ckpt = False
ckpt_file = 'MS_Model_4px24.pth' # for Kitti Dataset: 'KittiModel.pth'
# checkptname = "UCFModel"
checkptname = "MS_Model_4px_"



## Load input data

# set train list name:
trainFolderFile = './datasets/DisentanglingMotion/importing_data/moving_symbols/MovingSymbols2_trainlist.txt'
# trainFolderFile = 'trainlist01.txt'

# set training data directory:
rootDir = './datasets/DisentanglingMotion/importing_data/moving_symbols/output/MovingSymbols2_same_4px-OF/train'
# rootDir = './datasets/UCF-101-Frames'

trainFoldeList = getListOfFolders(trainFolderFile)[::10]
# if Kitti dataset: use listOfFolders instead of trainFoldeList
# listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]


trainingData = videoDataset(folderList=trainFoldeList,
                            rootDir=rootDir,
                            N_FRAME=N_FRAME)

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
model = OFModel(Drr, Dtheta, T, PRE, gpu_id)
model.cuda(gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1) # if Kitti: milestones=[100,150]
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
    scheduler.step()
    for i_batch, sample in enumerate(dataloader):

        data = sample['frames'].squeeze(0).cuda(gpu_id)
        expectedOut = Variable(data)
        inputData = Variable(data[:,0:FRA,:])

        optimizer.zero_grad()
        output = model.forward(inputData)

        if (px_ev):
            # Visualize time evolution of a single pixel from the input data
            v = inputData.data.cpu().numpy()
            px_id = 64*32+32
            px = v[:,:,px_id]

            if(px[px > 3.5].size > 2 and px[px < -3.5].size > 2 and count<1):

                fig, ax = plt.subplots(nrows=2, ncols=1)
                ax.flatten()
                ax[0].set_ylim([-5,5])
                ax[1].set_ylim([-5, 5])

                ax[0].plot(px[0, :])
                ax[1].plot(px[1, :])
                ax[0].set_title('Pixel Value for U')
                ax[1].set_title('Pixel Value for V')
                plt.savefig('pixel_evolution_OF.png')

                for i in range(FRA):

                    # inputData[:,:,px_id] = -10
                    hsv = np.zeros([64, 64, 3], dtype=np.float16)
                    hsv[:,:,0] = v[0,i,:].reshape(x_fra,y_fra)
                    hsv[:,:,2] = v[1,i,:].reshape(x_fra,y_fra)
                    hsv[32,32,1] = 5
                    #
                    # cv2.imwrite('inputOF_%5.2f.png' %i, hsv[:,:,0])
                    legend = np.zeros([2, 11, 3], dtype=np.float16)
                    for n in range(11):
                        legend[0, n, 0]= n-5
                        legend[1, n, 2] = n - 5
                    print('saving........')
                    scipy.misc.imsave('legend.png' ,legend)
                    scipy.misc.imsave('inputOF_%5.2f.png' %i, hsv)

                    # torchvision.utils.save_image(inputData[0, i, :].view(x_fra, y_fra), 'inputOF_U_F%5.2f.png' %i)
                    # torchvision.utils.save_image(inputData[1, i, :].view(x_fra, y_fra), 'inputOF_V_F%5.2f.png' %i)
                count = 1
                # sys.exit("Enough")

        # Visualize expected and output images.
        # po = output.data.cpu().numpy()
        # eo = expectedOut.data.cpu().numpy()
        # tmp1 = np.zeros([64, 64, 3], dtype=np.float16)
        # tmp1[:, :, 0] = po[0, FRA, :].reshape(x_fra, y_fra)
        # tmp1[:, :, 1] = po[1, FRA, :].reshape(x_fra, y_fra)
        #
        # scipy.misc.imsave('predicted_outputOF.png', tmp1)
        #
        # tmp2 = np.zeros([64, 64, 3], dtype=np.float16)
        # tmp2[:, :, 0] = eo[0, FRA, :].reshape(x_fra, y_fra)
        # tmp2[:, :, 1] = eo[1, FRA, :].reshape(x_fra, y_fra)
        #
        # scipy.misc.imsave('expected_outputOF.png', tmp2)

        # torchvision.utils.save_image(output[:, FRA].view(2,x_fra,y_fra), 'predicted_output.png',)
        # torchvision.utils.save_image(expectedOut[:, FRA].view(2,x_fra,y_fra), 'expected_output.png', )

        # Compute loss

        loss = loss_mse(output[:,FRA], expectedOut[:,FRA]) # if Kitti: loss = loss_mse(output, expectedOut)
        loss.backward()
        optimizer.step()
        loss_value.append(loss.data.item())

    loss_val = np.mean(np.array(loss_value))


    if epoch % saveEvery ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')

    if epoch % 4 == 0:
        print(model.state_dict()['l1.rr'])
        print(model.state_dict()['l1.theta'])
        # loss_val = float(loss_val/i_batch)
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val)