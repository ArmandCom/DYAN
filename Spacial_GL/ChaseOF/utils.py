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
from torch import nn
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
        self.x_fra = 240
        self.y_fra = 320
        self.numpixels = self.x_fra * self.y_fra

    def __len__(self):
        return len(self.listOfFolders)


    def readData(self, folderName):
        path = os.path.join(self.rootDir,folderName)
        OF = torch.FloatTensor(2,self.nfra,self.numpixels)
        OFori = torch.FloatTensor(1,2,self.nfra, self.x_fra, self.y_fra)
        for framenum in range(self.nfra):
            flow = np.load(os.path.join(path,str(framenum)+'.npy'))
            flow = np.transpose(flow,(2,0,1))
            OFori[:, :, framenum, :, :] = torch.from_numpy(flow).type(torch.FloatTensor).unsqueeze(0)
        flows = alignOF(OFori, self.nfra-1)
        if random.randint(1, 50) == 1:
            showOFs(flows, OFori, self.nfra, 'of')
        OF = flows.view((2, self.nfra, self.numpixels)).type(torch.FloatTensor)
        OFori = OFori.view((2, self.nfra, self.numpixels)).type(torch.FloatTensor)
        return OF, OFori

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame, FrameOri = self.readData(folderName)
        sample = { 'frames': Frame, 'framesori': FrameOri }

        return sample

def showOFs(flows, OFori, nfra, imname):
    x_fra = 240
    y_fra = 320
    OFal = flows[0,:,:,:,:]
    # OFal[:,0,0,:] = 1
    OFor = OFori[0,:,:,:,:]
    # OFor[:, 7, :, 30] = 1
    OFor[:,0,0,:] = 1
    tmp1 = torch.zeros((x_fra * nfra,  y_fra, 3))
    tmp2 = torch.zeros((x_fra * nfra, y_fra, 3))
    tmp1[:,:,0:2] = torch.cat(list(OFal.permute(1, 2, 3, 0)), 0)
    tmp2[:,:,0:2] = torch.cat(list(OFor.permute(1, 2, 3, 0)), 0)


    if nfra==1:
        tmp = torch.cat((tmp1, tmp2), 0)
        tmp = tmp.detach().numpy()
        scipy.misc.imsave('Pred_vs_Expected.png', tmp)
    else:
        tmp = torch.cat((tmp1, tmp2), 0)
        tmp = tmp.detach().numpy()
        scipy.misc.imsave('Align_vs_Ori_'+imname+'.png', tmp)
    # scipy.misc.imsave('originalOFs.png', tmp1)

    return

def showframes(frames, framesori, nfra, imname):
    x_fra = 240
    y_fra = 320

    framesal = frames[0,:,:,:,:]
    framesor = framesori[0,:,:,:,:]
    tmp1 = torch.zeros((x_fra * nfra,  y_fra, 3))
    tmp2 = torch.zeros((x_fra * nfra, y_fra, 3))
    tmp1 = torch.cat(list(framesal.permute(1, 2, 3, 0)), 0)
    tmp2 = torch.cat(list(framesor.permute(1, 2, 3, 0)), 0)


    if nfra==1:
        tmp = torch.cat((tmp1, tmp2), 0)
        tmp = tmp.detach().cpu().numpy()
        scipy.misc.imsave('Pred_vs_Expected'+imname+'.png', tmp)
    else:
        tmp = torch.cat((tmp1, tmp2), 1)
        tmp = tmp.detach().numpy()
        scipy.misc.imsave('Align_vs_Ori_'+imname+'.png', tmp)
    # scipy.misc.imsave('originalOFs.png', tmp1)

    return

def alignOF(flows, nfra):
    x_fra = 240
    y_fra = 320

    OFs = torch.zeros((1, 2, nfra+1, x_fra, y_fra))
    OFs[:, :, nfra, :, :] = flows[:, :, nfra, :, :] # for train
    for n in range(nfra): # to train range(nfra-1), ...flows, n, nfra-1
        OFs[:, :, n, :, :] = warpRecursive(flows[:, :, n, :, :], flows, n, nfra)
    return OFs

def alignOFtest(flows, nfra):
    # of0 = warpRecursive(flows[:, :, 0, :, :], flows, 0, 4)
    # of1 = warpRecursive(flows[:, :, 1, :, :], flows, 1, 4)
    # of2 = warpRecursive(flows[:, :, 2, :, :], flows, 2, 4)
    # of3 = warpRecursive(flows[:, :, 3, :, :], flows, 3, 4)
    # of4 = warpRecursive(flows[:, :, 4, :, :], flows, 4, 4)

    OFs = torch.zeros((1, 2, nfra, 128, 160))
    for n in range(nfra):
        OFs[:, :, n, :, :] = warpRecursive(flows[:, :, n, :, :], flows, n, nfra)
    return OFs

def warpRecursive(x, flows, ini, end):
    warped = x.clone() # cuidado
    for i in range(ini,end,1):
        tmp = warp(warped, flows[:, :, i, :, :])
        warped = tmp
    return warped

def warp(input,tensorFlow):

    torchHorizontal = torch.linspace(-1.0, 1.0, input.size(3))
    torchHorizontal = torchHorizontal.view(1, 1, 1, input.size(3)).expand(input.size(0), 1, input.size(2), input.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, input.size(2))
    torchVertical = torchVertical.view(1, 1, input.size(2), 1).expand(input.size(0), 1, input.size(2), input.size(3))

    tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1)
    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0) ], 1)
    # tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / 10, tensorFlow[:, 1:2, :, :] / 20], 1)

    # print(torch.mean(input-tensorFlow))
    return torch.nn.functional.grid_sample(input=input, grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

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
    data = data.str.split('/',expand=True)[1]
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

class videoDataset2stream(Dataset):
    """Dataset Class for Loading Video"""
    def __init__(self, folderList, rootDir, rootDir_rgb, N_FRAME): #N_FRAME = FRA+PRE

        """
        Args:
            N_FRAME (int) : Number of frames to be loaded
            rootDir (string): Directory with all the Frames/Videoes.
            Image Size = 240,320
            2 channels : U and V
        """
        self.listOfFolders = folderList
        self.rootDir = rootDir
        self.rootDir_rgb = rootDir_rgb
        self.nfra = N_FRAME
        # self.numpixels = 240*320
        self.x_fra = 240
        self.y_fra = 320
        self.numpixels = self.x_fra * self.y_fra

    def __len__(self):
        return len(self.listOfFolders)


    def readData(self, folderName):
        path = os.path.join(self.rootDir,folderName)
        pathrgb = os.path.join(self.rootDir_rgb,folderName)
        # OF = torch.FloatTensor(2,self.nfra,self.numpixels)
        OFori = torch.FloatTensor(1, 2, self.nfra, self.x_fra, self.y_fra)
        Frames = torch.FloatTensor(1, 3, self.nfra+1, self.x_fra, self.y_fra)
        alFrames = torch.FloatTensor(1, 3, self.nfra+1, self.x_fra, self.y_fra)
        for framenum in range(self.nfra):
            flow = np.load(os.path.join(path,str(framenum)+'.npy'))
            flow = np.transpose(flow,(2,0,1))
            OFori[:, :, framenum, :, :] = torch.from_numpy(flow).type(torch.FloatTensor).unsqueeze(0)

            img = np.load(os.path.join(pathrgb, str(framenum) + '.npy'))
            framenp = np.array(img)/255.
            framenp = np.transpose(framenp, (2, 0, 1))
            Frames[:, :, framenum, :, :] = torch.from_numpy(framenp).type(torch.FloatTensor).unsqueeze(0)
        img = np.load(os.path.join(pathrgb, str(self.nfra) + '.npy'))
        framenp = np.array(img) / 255.
        framenp = np.transpose(framenp, (2, 0, 1))
        Frames[:, :, self.nfra, :, :] = torch.from_numpy(framenp).type(torch.FloatTensor).unsqueeze(0)

        flows = alignOF(OFori, self.nfra-1)
        showOFs(flows, OFori, self.nfra, 'flows')

        alFrames[:, :, (self.nfra-1):(self.nfra+1), :, :] = Frames[:, :, (self.nfra-1):(self.nfra+1), :, :]
        for n in range(self.nfra-1):
            alFrames[:, :, n, :, :] = warpRecursive(Frames[:, :, n, :, :], flows, n, self.nfra-1)
        showframes(alFrames, Frames, self.nfra+1, 'frames')

        OF = flows.view((2, self.nfra, self.numpixels)).type(torch.FloatTensor)
        OFori = OFori.view((2, self.nfra, self.numpixels)).type(torch.FloatTensor)
        Frames = Frames.view((3, self.nfra+1, self.numpixels)).type(torch.FloatTensor)
        alFrames = alFrames.view((3, self.nfra+1, self.numpixels)).type(torch.FloatTensor)

        return OF, OFori, alFrames, Frames

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        of, ofori, alframes, frames = self.readData(folderName)
        sample = { 'of': of, 'ofori': ofori, 'alframes': alframes, 'frames': frames }

        return sample

class videoDataset3ch(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, folderList, rootDir, N_FRAME):  # N_FRAME = FRA+PRE

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
        self.x_fra = 240
        self.y_fra = 320
        self.numpixels = self.x_fra * self.y_fra

    def __len__(self):
        return len(self.listOfFolders)

    def readData(self, folderName):
        path = os.path.join(self.rootDir, folderName)
        OF = torch.FloatTensor(2, self.nfra, self.numpixels)
        OFori = torch.FloatTensor(1, 2, self.nfra, self.x_fra, self.y_fra)
        for framenum in range(self.nfra):
            flow = np.load(os.path.join(path, str(framenum) + '.npy'))
            flow = np.transpose(flow, (2, 0, 1))
            OFori[:, :, framenum, :, :] = torch.from_numpy(flow).type(torch.FloatTensor).unsqueeze(0)
        flows = alignOF(OFori, self.nfra - 1)
        showOFs(flows, OFori, self.nfra, 'of')
        OF = flows.view((2, self.nfra, self.numpixels)).type(torch.FloatTensor)
        OFori = OFori.view((2, self.nfra, self.numpixels)).type(torch.FloatTensor)
        return OF, OFori

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame, FrameOri = self.readData(folderName)
        sample = {'frames': Frame, 'framesori': FrameOri}

        return sample

    # def warpRecursive(x, flows, ini, end):
    #     warped = x.clone()  # cuidado
    #     for i in range(ini, end, 1):
    #         tmp = warp(warped, flows[:, :, i, :, :])
    #         warped = tmp
    #     return warped

    # def alignOF(flows, nfra):
    #
    #     OFs = torch.zeros((1, 2, nfra + 1, 128, 160))
    #     OFs[:, :, nfra, :, :] = flows[:, :, nfra, :, :]  # for train
    #     for n in range(nfra):  # to train range(nfra-1), ...flows, n, nfra-1
    #         OFs[:, :, n, :, :] = warpRecursive(flows[:, :, n, :, :], flows, n, nfra)
    #     return OFs