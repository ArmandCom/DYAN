############################# Import Section #################################
## Generic imports
import os
import time
import math
import random
import numpy as np
import pandas as pd
import scipy
import sys
from PIL import Image
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from skimage import measure
############################# Import Section #################################

## Dataloader for PyTorch.
class videoDataset(Dataset):
    """Dataset Class for Loading Video"""
    def __init__(self, folderList, rootDir, N_FRAME): #N_FRAME = FRA+PRE

        """
        Args:
            N_FRAME (int) : Number of frames to be loaded
            rootDir (string): Directory with all the Frames/Videoes.
            Image Size = 240,320
            2 channels : U and V
        """
        self.listOfFolders = folderList
        self.rootDir = rootDir
        self.nfra = N_FRAME
        # self.numpixels = 240*320 # If Kitti dataset, self.numpixels = 128*160
        self.numpixels = 128*160 # MNIST moving symbols dataset

    def __len__(self):
        return len(self.listOfFolders)


    def readData(self, folderName):
        path = os.path.join(self.rootDir,folderName)
        OF = torch.FloatTensor(2,self.nfra,self.numpixels)
        OFori = np.zeros((2,self.nfra,128,160))
        for framenum in range(self.nfra):

            flow = np.load(os.path.join(path,str(framenum)+'.npy'))
            flow = np.transpose(flow,(2,0,1))
            OFori[:, framenum, :, :] = flow


        flows = alignOF(OFori, self.nfra)
        print(np.mean(flows[:,4,:,:])/np.mean(OFori[:,4,:,:]))
        showOFs(flows, OFori)
        OF = torch.from_numpy(flows.reshape(2, self.nfra, self.numpixels)).type(torch.FloatTensor)
        OF_ori = torch.from_numpy(OFori.reshape(2, self.nfra, self.numpixels)).type(torch.FloatTensor)
        return OF#, OF_ori

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame = self.readData(folderName)
        sample = { 'frames': Frame }

        return sample

def showOFs(flows, OFori):

    tmp1 = np.zeros([128 * 5,  160, 3], dtype=np.float16)
    tmp2 = np.zeros([128 * 5, 160, 3], dtype=np.float16)
    tmp = np.zeros([128 * 5, 160*2, 3], dtype=np.float16)
    tmp1[:,:,0:2] = np.concatenate(list(np.transpose(flows, (1, 2, 3, 0))), axis=0)
    tmp2[:,:,0:2] = np.concatenate(list(np.transpose(OFori, (1, 2, 3, 0))), axis=0)
    tmp = np.concatenate((tmp1, tmp2), axis=1)

    scipy.misc.imsave('AligvsOriOFs.png', tmp)
    # scipy.misc.imsave('originalOFs.png', tmp1)

    return

def alignOF(flows, nfra):

    of00 = warp(flows[:, 0, :, :], flows[:, 0, :, :])
    of001 = warp(of00, flows[:, 1, :, :])
    of0012 = warp(of001, flows[:, 2, :, :])
    of00123 = warp(of0012, flows[:, 3, :, :])
    of001234 = warp(of00123, flows[:, 4, :, :])

    # print(np.amax(flows[:,0,:,:]-of001))
    # print(np.amax(flows[:, 0, :, :]))

    of11 = warp(flows[:, 1, :, :], flows[:, 1, :, :])
    of112 = warp(of11, flows[:, 2, :, :])
    of1123 = warp(of112, flows[:, 3, :, :])
    of11234 = warp(of1123, flows[:, 4, :, :])

    of22 = warp(flows[:, 2, :, :], flows[:, 2, :, :])
    of223 = warp(of22, flows[:, 3, :, :])
    of2234 = warp(of223, flows[:, 4, :, :])

    of33 = warp(flows[:, 3, :, :], flows[:, 3, :, :])
    of334 = warp(of33, flows[:, 4, :, :])

    of44 = warp(flows[:, 4, :, :], flows[:, 4, :, :])

    OFs = np.zeros((2,nfra,128,160))
    np.stack((of001234, of11234, of2234, of334, of44), axis=1, out=OFs)


    # f = {}
    # for i in range(nfra):
    #     f["of" + str(i) + str(i)] = warp(flows[:, i, :, :], flows[:, i, :, :])

    return OFs

def warp(x, flow):
    flow_back = scipwarp(x, flow[0, :, :], flow[1, :, :])
    # flow_back = np.clip(flow_back, 0, 1.)
    return flow_back

def scipwarp(flow, u, v):

    _, M, N = flow.shape
    x = np.linspace(0,N-1, N)
    y = np.linspace(0,M-1, M)
    x, y = np.meshgrid(x, y)

    x += u
    y += v
    warped = np.zeros(flow.shape)
    warped[0, :, :] = scipy.ndimage.map_coordinates(flow[0, :, :], [y.ravel(), x.ravel()], order=1,
                                                    mode='nearest').reshape(flow.shape[1], flow.shape[2])
    warped[1, :, :] = scipy.ndimage.map_coordinates(flow[1, :, :], [y.ravel(), x.ravel()], order=1,
                                                    mode='nearest').reshape(flow.shape[1], flow.shape[2])
    return warped




## Design poles
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


def PSNR(predi, pix):
    pix = pix.astype(float)
    predict = predi.numpy().astype(float)
    mm = np.amax(predict)
    mse = np.linalg.norm(predict - pix)
    mse = mse / (256 * 256)
    psnr = 10 * math.log10(mm ** 2 / mse)

    return psnr


def SSIM(predi, pix):
    pix = pix.astype(float)
    predict = predi.numpy().astype(float)
    ssim_score = measure.compare_ssim(pix[:, :], predict[:, :], win_size=13, data_range=255,
                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                      K1=0.01, K2=0.03)

    return ssim_score


# Sharpness
def SHARP(predict, pix):
    predict = predict
    pix = torch.from_numpy(pix).float()
    s1 = pix.size()[0]
    s2 = s1 + 1  # dim after padding
    pp1 = torch.cat((torch.zeros(1, s1), predict), 0)  # predict top row padding
    pp2 = torch.cat((torch.zeros(s1, 1), predict), 1)  # predict first col padding
    oo1 = torch.cat((torch.zeros(1, s1), pix), 0)  # pix top row padding
    oo2 = torch.cat((torch.zeros(s1, 1), pix), 1)  # pix first col padding

    dxpp = torch.abs(pp1[1:s2, :] - pp1[0:s1, :])
    dypp = torch.abs(pp2[:, 1:s2] - pp2[:, 0:s1])
    dxoo = torch.abs(oo1[1:s2, :] - oo1[0:s1, :])
    dyoo = torch.abs(oo2[:, 1:s2] - oo2[:, 0:s1])

    gra = torch.sum(torch.abs(dxoo + dyoo - dxpp - dypp))
    mm = torch.max(predict)

    gra = gra / (256 * 256)

    gra = (mm ** 2) / gra

    sharpness = 10 * math.log10(gra)

    return sharpness