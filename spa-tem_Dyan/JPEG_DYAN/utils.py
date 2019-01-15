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
import cv2
## Imports related to PyTorch
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc # pip install Pillow
import matplotlib.pylab as pylab

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc # pip install Pillow
############################# Import Section #################################

## Dataloader for PyTorch.
class videoDataset(Dataset):
    """Dataset Class for Loading Video"""
    def __init__(self, folderList, rootDir, blockSize, nfra): #N_FRAME = FRA+PRE

        """
        Args:
            N_FRAME (int) : Number of frames to be loaded
            rootDir (string): Directory with all the Frames/Videoes.
            Image Size = 240,320
            2 channels : U and V
        """
        self.listOfFolders = folderList
        self.rootDir = rootDir
        self.numpixels = 640*480 # Kitti
        self.x_fra = 480
        self.y_fra = 640
        self.nfra = nfra
        self.blockSize = blockSize
        self.nchan = 3

    def __len__(self):
        return len(self.listOfFolders)


    def readData(self, folderName):


        path = os.path.join(self.rootDir,folderName)
        OF = torch.FloatTensor(self.nchan, self.nfra, self.numpixels)
        flows = np.zeros((self.nfra, self.nchan, self.x_fra, self.y_fra))

        '''Caltech'''
        frames = [each for each in os.listdir(os.path.join(self.rootDir, folderName)) if
                  each.endswith(('.jpg', '.jpeg', '.bmp', 'png'))]
        frames.sort()
        for framenum in range(self.nfra): #add random value to add to ini

            '''KittiFlows'''
            # flow = np.load(os.path.join(path,str(framenum)+'.npy')).astype(float)
            '''Caltech'''
            flowname = os.path.join(path, frames[framenum])
            # print(flowname)
            flow = misc.imread(flowname).astype(float)/255


            flowsize = flow.shape
            dct = np.zeros(flowsize).astype(float)

            # Do 8x8 DCT on image (in-place)
            for i in r_[:flowsize[0]:self.blockSize]:
                for j in r_[:flowsize[1]:self.blockSize]:

                    dct[i:(i + self.blockSize), j:(j + self.blockSize)] = \
                        dct2(flow[i:(i + self.blockSize), j:(j + self.blockSize)])

                    # dct[i:(i + self.blockSize), j:(j + self.blockSize)] = \
                    #     np.fft.fft2(flow[i:(i + self.blockSize), j:(j + self.blockSize)])

            dct = np.transpose(dct, (2, 0, 1))
            flows[framenum, :, :, :] = np.transpose(flow, (2,0,1))


            '''CHECK SIZES'''
            OF[:,framenum] = torch.from_numpy(dct.reshape(self.nchan,self.numpixels)).type(torch.FloatTensor)
        return OF, flows


    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame, Flows = self.readData(folderName)
        sample = { 'frames': Frame, 'flows': Flows }

        return sample


## Design poles

def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct( a, axis=0, norm='ortho', type=2 ), axis=1, norm='ortho', type=2 )

def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct( a, axis=0 , norm='ortho', type=2), axis=1 , norm='ortho', type=2)


def gridRing(N):
    epsilon_low = 0.25
    epsilon_high = 0.15
    rmin = (1-epsilon_low)
    rmax = (1+epsilon_high)
    thetaMin = 0.001
    thetaMax = np.pi/2 - 0.001
    delta = 0.001
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


def getListOfFolders(File):
    data = pd.read_csv(File, sep=" ", header=None)[0]
    data = data.str.split('/',expand=True)[1] # --> This is commented for the warp.py!!! review for other scripts
    data = data.str.rstrip(".avi").values.tolist()

    return data

def getListOfFolders_warp(File):
    data = pd.read_csv(File, sep=" ", header=None)[0]
    data = data.str.rstrip(".avi").values.tolist()

    return data

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