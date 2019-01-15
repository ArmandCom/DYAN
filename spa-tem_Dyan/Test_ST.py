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
import sys
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from skimage import io, transform
from matplotlib.backends.backend_pdf import PdfPages

## Dependencies classes and functions
from utils import PSNR
from utils import SHARP
from utils import asMinutes
from utils import timeSince
from utils import getListOfFolders
#from utils import creatRealDictionary
from DyanOFST_SW import creatRealDictionary
from skimage import measure
from DyanOFST_SW import OFModel

from scipy.misc import imread, imresize
from skimage.measure import compare_mse as mse
from collections import defaultdict
import matplotlib
import scipy.io as sio
import cv2
from matplotlib.colors import hsv_to_rgb
############################# Import Section #################################
import math
from pylab import imshow, show, get_cmap
import scipy
import math
start = time.time()
from util.pyflow import pyflow
# Hyper Parameters
FRA = 9
PRE = 1
N_FRAME = FRA+PRE
T = FRA
gpu_id = 1
x_fra = 128
y_fra = 160


def loadModel(ckpt_file):
    loadedcheckpoint = torch.load(ckpt_file)  # ,map_location={'cuda:1':'cuda:0'})
    stateDict = loadedcheckpoint['state_dict']

    Wx = 32
    Wy = 40
    # load parameters
    DthetaSH = stateDict['esH.theta']
    DrrSH = stateDict['esH.rr']
    DthetaSV = stateDict['esV.theta']
    DrrSV = stateDict['esV.rr']
    DthetaT = stateDict['et.theta']
    DrrT = stateDict['et.rr']

    model = OFModel(DrrT, DthetaT, DrrSH, DthetaSH, DrrSV, DthetaSV, T, PRE, x_fra, y_fra)
    model.cuda(gpu_id)

    DrrT = Variable(DrrT.cuda(gpu_id))
    DthetaT = Variable(DthetaT.cuda(gpu_id))
    DrrSH = Variable(DrrSH.cuda(gpu_id))
    DthetaSH = Variable(DthetaSH.cuda(gpu_id))
    DrrSV = Variable(DrrSV.cuda(gpu_id))
    DthetaSV = Variable(DthetaSV.cuda(gpu_id))

    dicT = creatRealDictionary(N_FRAME, DrrT, DthetaT, gpu_id)
    dicSH = creatRealDictionary(Wx, DrrSH, DthetaSH, gpu_id)
    dicSV = creatRealDictionary(Wy, DrrSV, DthetaSV, gpu_id)

    return model, DrrT, DthetaT, DrrSH, DthetaSH, DrrSV, DthetaSV, dicT, dicSH, dicSV


def process_im(im, desired_sz=(128, 160)):
    target_ds = float(desired_sz[0]) / im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d + desired_sz[1]]
    # im = imutils.resize(im, width=160,height=128)
    return im


def SSIM(predi, pix):
    pix = pix.astype(float)
    predict = predi.astype(float)
    ssim_score = measure.compare_ssim(pix, predict, win_size=11, data_range=1., multichannel=True,
                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                      K1=0.01, K2=0.03)

    return ssim_score


def scipwarp(img, u, v):
    M, N, _ = img.shape
    x = np.linspace(0, N - 1, N)
    y = np.linspace(0, M - 1, M)
    x, y = np.meshgrid(x, y)
    x += u
    y += v
    warped = img
    warped[:, :, 0] = scipy.ndimage.map_coordinates(img[:, :, 0], [y.ravel(), x.ravel()], order=1,
                                                    mode='nearest').reshape(img.shape[0], img.shape[1])
    warped[:, :, 1] = scipy.ndimage.map_coordinates(img[:, :, 1], [y.ravel(), x.ravel()], order=1,
                                                    mode='nearest').reshape(img.shape[0], img.shape[1])
    warped[:, :, 2] = scipy.ndimage.map_coordinates(img[:, :, 2], [y.ravel(), x.ravel()], order=1,
                                                    mode='nearest').reshape(img.shape[0], img.shape[1])
    return warped

alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0

import time
current_milli_time = lambda: int(round(time.time() * 1000))

gpu_id = 1
ckpt_file = '/home/armandcomas/DYAN/preTrainedModel/Kitti_Normal-ST-test18.pth'
rootDir = '/home/armandcomas/datasets/Caltech/images'

## Load model from a checkpoint file

folderList = ['set10V011'] # set10V011
__imgsize__ = (128,160)
mse = []
ssim = []
psnr = []
c_list = []

for folder in folderList:
    print(folder)
    frames = [each for each in os.listdir(os.path.join(rootDir, folder)) if
              each.endswith(('.jpg', '.jpeg', '.bmp', 'png'))]
    frames.sort()
    print(len(frames))
    for i in range(25, 100, 1):
        sample = torch.FloatTensor(2, N_FRAME - 1, 128 * 160)
        model, DrrT, DthetaT, DrrSH, DthetaSH, DrrSV, DthetaSV, dicT, dicSH, dicSV = loadModel(ckpt_file)
        print(i)
        for ii in range(i, FRA + i):  # for INCR WNDW range(25,FRA+i)
            imgname = os.path.join(rootDir, folder, frames[ii])
            img = Image.open(imgname)
            img1 = process_im(np.array(img)) / 255.

            imgname = os.path.join(rootDir, folder, frames[ii + 1])
            img = Image.open(imgname)
            img2 = process_im(np.array(img)) / 255.

            u, v, _ = pyflow.coarse2fine_flow(img2, img1, alpha, ratio, minWidth,
                                              nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)

            # u = cv2.GaussianBlur(u,(3,3),0)
            # v = cv2.GaussianBlur(v,(3,3),0) #'Doesn't work with the poles trained for regular OF
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)
            flow = np.transpose(flow, (2, 0, 1))

            sample[:, ii - i, :] = torch.from_numpy(flow.reshape(2, 128 * 160)).type(
                torch.FloatTensor)  # for INCR WNDW sample[:,ii-25,:]

        # sample[0,0,10290] = 2
        img_plt = sample[0, 0, :].cpu().numpy().reshape(128, 160)

        plt.imshow(img_plt, cmap='gray')
        # plt.show()

        imgname = os.path.join(rootDir, folder, frames[ii + 2])
        img = Image.open(imgname)
        original = process_im(np.array(img)) / 255.

        # plt.imshow(original, cmap='gray')
        # plt.show()

        imgname = os.path.join(rootDir, folder, frames[ii + 1])
        img = Image.open(imgname)
        tenth = process_im(np.array(img)) / 255.

        inputData = sample.cuda(gpu_id)
        # start = current_milli_time()

        with torch.no_grad():
            sparse = model.forwardE(Variable(inputData))
            c = sparse.cpu().numpy()
            c_list.append(c)

        ## INCREASING WINDOW
        # N_FRAME += 1
        # T += 1

        [predictionH, predictionV] = model.forwardD(sparse)
        prediction = (predictionH.detach().cpu().numpy() + predictionV.detach().cpu().numpy()) / 2
        img_back = scipwarp(tenth, prediction[0, :, :], prediction[1, :, :])
        img_back = np.clip(img_back, 0, 1.)

        # plt.imshow(img_back, cmap='gray')
        # plt.show()

        meanserror = np.mean((img_back - original) ** 2)
        mse.append(meanserror)
        peaksnr = 10 * math.log10(1. / meanserror)
        psnr.append(peaksnr)
        ssim.append(SSIM(original, img_back))

    print('Mse: ', np.mean(np.array(mse)))
    print('PSNR: ', np.mean(np.array(psnr)))
    print('SSIM: ', np.mean(np.array(ssim)))

    print('done!')