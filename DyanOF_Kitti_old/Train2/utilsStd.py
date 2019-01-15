## Author - Wen

############################# Import Section #################################
## Generic imports
import os
import time
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
## Imports related to PyTorch
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import measure
import cv2
############################# Import Section #################################


## Dataloader for PyTorch.
class videoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, listOfFolders, rootDir):

        """
        Args:
            N_FRAME (int) : Number of frames to be loaded
            rootDir (string): Directory with all the Frames/Videoes.
        """
        self.listOfFolders = listOfFolders
        self.rootDir = rootDir


    def __len__(self):
        return len(self.listOfFolders)

    def readData(self, folderName):
        path = os.path.join(self.rootDir,folderName)
        numBatches = 1#min(int(nFrames/nfra),100)
        sample = torch.FloatTensor(numBatches,2,10,128*160)
        for batchnum in range(numBatches):
            for framenum in range(10):

                flow = np.load(os.path.join(path,str(framenum)+'.npy'))
                flow = np.transpose(flow,(2,0,1))
                sample[batchnum,:,framenum,:] = torch.from_numpy(flow.reshape(2,128*160)).type(torch.FloatTensor)

        return sample

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        frames = self.readData(folderName)
        sample = { 'frames': frames }

        return sample


## Design poles

def gridRing(N):
    epsilon_low = 0.20
    epsilon_high = 0.20
    rmin = (1-epsilon_low)
    rmax = (1+epsilon_high)
    thetaMin = 0.001
    thetaMax = np.pi/2 - 0.001
    delta = 0.05
    Npole = int(N/4)
    Pool = generateGridPoles(delta,rmin,rmax,thetaMin,thetaMax)
    M = len(Pool)
    idx = random.sample(range(0, M), Npole)
    P = Pool[idx]
    Pall = np.concatenate((P,-P, np.conjugate(P),np.conjugate(-P)),axis = 0)

    return P,Pall

## Generate the grid on poles
def generateGridPoles(delta,rmin,rmax,thetaMin,thetaMax):
    rmin2 = pow(rmin,2)
    rmax2 = pow(rmax,2)
    xv = np.arange(-rmax,rmax,delta)
    x,y = np.meshgrid(xv,xv,sparse = False)
    mask = np.logical_and( np.logical_and(x**2 + y**2 >= rmin2 , x**2 + y **2 <= rmax2),
                           np.logical_and(np.angle(x+1j*y)>=thetaMin, np.angle(x+1j*y)<=thetaMax ))
    px = x[mask]
    py = y[mask]
    P = px + 1j*py

    return P

def creatRealDictionary(T,Drr,Dtheta):
    n = Drr.shape[0]
    WVar = []
    # Wones = torch.ones(1).cuda(1)
    # Wones  = Variable(Wones,requires_grad=False)

    for i in range(0,T):

        W1 = torch.mul(torch.pow(Drr,i) , torch.cos(i * Dtheta))
        W2 = torch.mul ( torch.pow(-Drr,i) , torch.cos(i * Dtheta) )
        W3 = torch.mul ( torch.pow(Drr,i) , torch.sin(i *Dtheta) )
        W4 = torch.mul ( torch.pow(-Drr,i) , torch.sin(i*Dtheta) )
        W = torch.cat((W1, W2, W3, W4), 0)
        WVar.append(W.view(1,-1))

    dic = torch.cat((WVar),0)

    mean_val = torch.mean(dic, 0).unsqueeze(0).data
    dic = (dic - mean_val)
    std = dic.std(0).unsqueeze(0).data
    std[std == 0] = 1
    dic = dic / std
    # G = torch.norm(dic,p=2,dim=0)
    # idx = (G == 0).nonzero()
    # nG = G.clone()
    # nG[idx] = np.sqrt(T)
    # G = nG
    # dic = dic/G
    return dic


# Create Gamma for Fista
def getWeights(Pall,N):
    g2 = pow(abs(Pall),2)
    g2N = np.power(g2,N)

    GNum = 1-g2
    GDen = 1-g2N
    idx = np.where(GNum == 0)[0]

    GNum[idx] = N
    GDen[idx] = pow(N,2)
    G = np.sqrt(GNum/GDen)
    return np.concatenate((np.array([1]),G))

## Functions for printing time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

## Function to save the checkpoint
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


# PSNR = 10 lg (max(pred)^2 / MSE)

def PSNR(predi,pix):
    pix = pix.astype(float)
    predict = predi.numpy().astype(float)
    mm = np.amax(predict)
    mse = np.linalg.norm(predict - pix)
    mse = mse/(256*256)
    psnr = 10* math.log10(mm **2/mse)

    return psnr

def SSIM(predi,pix):
    pix = pix.astype(float)
    predict = predi.numpy().astype(float)
    ssim_score = measure.compare_ssim(pix[:,:], predict[:,:], win_size=13, data_range = 255,
                  gaussian_weights=True,sigma = 1.5,use_sample_covariance=False,
                  K1=0.01,K2=0.03)

    return ssim_score

# Sharpness
def SHARP(predict,pix):
    predict = predict
    pix = torch.from_numpy(pix).float()
    s1 = pix.size()[0]
    s2 = s1+1 # dim after padding
    pp1 = torch.cat((torch.zeros(1,s1),predict),0) # predict top row padding
    pp2 = torch.cat((torch.zeros(s1,1),predict),1) # predict first col padding
    oo1 = torch.cat((torch.zeros(1,s1),pix),0) # pix top row padding
    oo2 = torch.cat((torch.zeros(s1,1),pix),1) # pix first col padding

    dxpp = torch.abs( pp1[1:s2,:] - pp1[0:s1,:] )
    dypp = torch.abs( pp2[:,1:s2] - pp2[:,0:s1] )
    dxoo = torch.abs( oo1[1:s2,:] - oo1[0:s1,:] )
    dyoo = torch.abs( oo2[:,1:s2] - oo2[:,0:s1] )

    gra = torch.sum ( torch.abs(dxoo+dyoo-dxpp-dypp) )
    mm = torch.max(predict)

    gra = gra/(256*256)

    gra = (mm**2) / gra

    sharpness = 10* math.log10(gra)

    return sharpness

def getListOfFolders(File):
    data = pd.read_csv(File, sep=" ", header=None)[0]
    data = data.str.split('/',expand=True)[1]
    data = data.str.rstrip(".avi").values.tolist()

    return data

def getIndex(sources_list):
    index = {}
    folderName = set(sources_list)
    for folder in folderName:
        first = 0
        last = 0
        for count, name in enumerate(sources_list):
            if folder == name:
                if first == 0:
                    if count == 0:
                        first = count-1
                    else:
                        first = count

                last = count
                index[folder] = {"first":first, "last":last+1}
            else:
                continue

    if index['city-2011_09_26_drive_0001_sync']['first'] == -1:
        index['city-2011_09_26_drive_0001_sync']['first'] = 0

    return index
