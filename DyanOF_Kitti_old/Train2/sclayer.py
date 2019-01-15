## Author - Wen

############################# Import Section #################################

## Imports related to PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

## Generic imports
import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from skimage import io, transform
from matplotlib.backends.backend_pdf import PdfPages

## Dependencies classes and functions
from utils import gridRing
from utils import asMinutes
from utils import timeSince
from utils import getWeights
from utils import getIndex
from utils import videoDataset
from utils import save_checkpoint
from utils import getListOfFolders
from utils import generateGridPoles
from utils import creatRealDictionary


## Import Model
from sc2layerModelStd import SC2
from sc2layerModelStd import SClayer

############################# Import Section #################################


checkptname = "Kitti_stdNN_Normal_lam0.1_"
## HyperParameters for the Network
num_of_poles = 40
EPOCH = 150
BATCH_SIZE = 1
LR = 0.0001				# Learning rate
DOWNLOAD_MNIST = False
print_every = 30
FRA = 9					# Training sequnece length
PRE = 1						# Prediction sequence length
N_FRAME = FRA+PRE
N = num_of_poles*4 			# Number of poles in dic
K = 5 						# PCA output atom size
T = FRA 					# Hankel length

x_sz = 64
y_sz = 64

## Initializing r, theta and Gamma
P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

Gamma = getWeights(Pall,T)
Gamma = torch.from_numpy(Gamma).float()

gpu_id = 1
rootDir = '/home/armandcomas/datasets/Kitti_Flows/'
# rootDir = '/home/armandcomas/DYAN/Code/datasets/DisentanglingMotion/importing_data/moving_symbols/output/MovingSymbols2_same_4px-OF/train'

listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]
# listOfFolders_1[0] = os.path.join(rootDir,listOfFolders_1[0])
# listOfFolders_1[1] = os.path.join(rootDir,listOfFolders_1[1])
# listOfFolders_H = [name for name in os.listdir(listOfFolders_1[0]) if os.path.isdir(os.path.join(listOfFolders_1[0], name))]
# listOfFolders_V = [name for name in os.listdir(listOfFolders_1[1]) if os.path.isdir(os.path.join(listOfFolders_1[1], name))]



# listFolderFile = '/home/armandcomas/DYAN/Code/datasets/DisentanglingMotion/importing_data/moving_symbols/MovingSymbols2_trainlist.txt'
# listOfFolders = getListOfFolders(listFolderFile)[::10]

# Function for the PyTorch Dataloader
trainingData = videoDataset(listOfFolders=listOfFolders,
                            rootDir=rootDir)

dataloader = DataLoader(trainingData, 
                        batch_size=BATCH_SIZE ,
                        shuffle=True, num_workers=1)


## Create the model
model = SC2(Drr, Dtheta, Gamma, T, PRE)
model.cuda(gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.5)
loss_l1 = nn.L1Loss()
loss_mse = nn.MSELoss()
start_epoch = 1
ckpt_file = 'Kitti_stdNN_Normal_lam0.1_30.pth'
load_ckpt = False
## If want to continue training from a checkpoint
if(load_ckpt):
    loadedcheckpoint = torch.load(ckpt_file)
    start_epoch = loadedcheckpoint['epoch']
    model.load_state_dict(loadedcheckpoint['state_dict'])
    optimizer.load_state_dict(loadedcheckpoint['optimizer'])


print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()


## Start the Training

for epoch in range(start_epoch, EPOCH+1):
    loss_value = []
    loss_value_rec = []
    # exp_lr_scheduler.step()
    for i_batch, sample in enumerate(dataloader):
        dataBatch = sample['frames'].squeeze(0)
        numBatches = dataBatch.shape[0]
        los_val = []
        loss_val_rec = []
        for batchnum in range(numBatches):
            data = dataBatch[batchnum].cuda(gpu_id)
            expectedOut = data
            inputData = data[:,0:FRA,:]
            # print(torch.max(inputData),torch.min(inputData))
            # inputData = inputData
            # print(inputData.shape)
            optimizer.zero_grad()
            output = model.forward(inputData)

            # torchvision.utils.save_image(output[:, FRA].view(2, x_sz, y_sz), 'predicted_output.png', )
            # torchvision.utils.save_image(expectedOut[:, FRA].view(2, x_sz, y_sz), 'expected_output.png', )


            loss = loss_mse(output[:,FRA,:], expectedOut[:,FRA,:])
            lossR = loss_mse(output[:, 0:FRA, :], expectedOut[:, 0:FRA, :])
            lossR.backward()
            optimizer.step()
            los_val.append(loss.data[0])
            loss_val_rec.append(lossR.data[0])
        loss_value.append(np.mean(np.array(los_val)))
        loss_value_rec.append(np.mean(np.array(loss_val_rec)))
    loss_val = np.mean(np.array(loss_value))
    loss_val_rec = np.mean(np.array(loss_value_rec))
    if epoch % print_every ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')

    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| train loss: %.4f' % loss_val_rec)
