############################# Import Section #################################
## Imports related to PyTorch
import torch
from torch.autograd import Variable

## Generic imports
import os
import time
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt


from DyanOF import OFModel
from DyanOF import creatRealDictionary
# from utils import getListOfFolders

from skimage import measure
import scipy
from scipy.misc import imread, imresize
from utils import *
# from pyflow import pyflow
############################# Import Section #################################

# TODO revisar que es facin servir sempre els mateixos pols i que siguin els correctes
# TODO revisar que no s'aprengui res

gpu_id = 2
ckpt_file = '/home/armandcomas/DYAN/preTrainedModel/' \
              'Kitti_ChaseOF_normDic-stdx_lam06_lossFuFRAPREweighted_FRA6-PRE1_Comp-ori_200.pth'
rootDir = '/home/armandcomas/datasets/Caltech/images'
flowDir = '/home/armandcomas/datasets/Caltech_flows/flowsall'

# Hyper Parameters
FRA = 6
PRE = 1
FPRE = 1
N_FRAME = FRA+PRE
T = FRA
numOfPixels = 128*160
lam = 0.6

def loadModel(ckpt_file):
    loadedcheckpoint = torch.load(ckpt_file)
    #model.load_state_dict(loadedcheckpoint['state_dict'])
    #optimizer.load_state_dict(loadedcheckpoint['optimizer'])
    stateDict = loadedcheckpoint['state_dict']

    # load parameters
    Dtheta = stateDict['l1.theta']
    Drr    = stateDict['l1.rr']
    model = OFModel(Drr, Dtheta, T, PRE, lam, gpu_id)
    model.cuda(gpu_id)
    Drr = Variable(Drr.cuda(gpu_id))
    Dtheta = Variable(Dtheta.cuda(gpu_id))
    dictionary = creatRealDictionary(N_FRAME,Drr,Dtheta, gpu_id)

    return model, dictionary, Drr, Dtheta

def process_im(im, desired_sz=(128, 160)):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    # im = imutils.resize(im, width=160,height=128)
    return im


def SSIM(predi,pix):
    pix = pix.astype(float)
    predict = predi.astype(float)
    ssim_score = measure.compare_ssim(pix, predict, win_size=11, data_range = 1.,multichannel=True,
                    gaussian_weights=True,sigma = 1.5,use_sample_covariance=False,
                    K1=0.01,K2=0.03)

    return ssim_score


############################################################################

## Load the model
ofmodel, dictionary, Drr, Dtheta = loadModel(ckpt_file)
ofSample = torch.FloatTensor(2, FRA+PRE+FPRE, numOfPixels)
ofSampleF = torch.FloatTensor(2, FRA+PRE, numOfPixels)

folderList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
folderList.sort()

##################### Testing script ONLY for Kitti dataset: ##############
fomse=[]
fossim=[]
fomsef=[]
fossimf=[]

for folder in folderList:
    print("Started testing for - "+ folder)


    frames = [each for each in os.listdir(os.path.join(rootDir, folder)) if each.endswith(('.jpeg','png','jpg'))]
    frames.sort()

    for i in range(0,len(frames)-(FRA+PRE+FPRE+1),FRA+PRE+FPRE):
        #i in range(1):
        #i in range(0,len(frames)-11,10):
        #print('starting from 0?',i)
        
        mse = []
        ssim = []
        ssimf = []
        msef = []

        for k in range(FRA+PRE+FPRE):
                flow = np.load(os.path.join(flowDir,folder,str(k+i)+'.npy'))
                flow = np.transpose(flow,(2,0,1))
                ofSample[:,k,:] = torch.from_numpy(flow.reshape(2,numOfPixels)).type(torch.FloatTensor)


        alSample = alignOFtest(ofSample[:,0:FRA,:].view(2, FRA, 128, 160).unsqueeze(0), FRA)
        showOFs(alSample, ofSample[:,0:FRA,:].view(2, FRA, 128, 160).unsqueeze(0), FRA, '')
        ofinputData = alSample.cuda(gpu_id)

        with torch.no_grad():
            ofprediction = ofmodel.forward(Variable(ofinputData).view((2, FRA, 128 * 160)))
        #print('out put from DYAN is in size of:',ofprediction.shape)
        sample_pred = torch.zeros((1,2,FRA+PRE,128,160))
        sample_pred[:,:,0:FRA,:,:] = alSample.clone().cpu()
        sample_pred[:,:,FRA,:,:] = ofprediction[:,FRA,:].view(2,128,160).unsqueeze(0).cpu()

        showOFs(sample_pred,
                ofprediction[:, 0:(FRA+PRE), :].view(2, FRA+PRE, 128, 160).unsqueeze(0).cpu(), FRA+PRE, 'alignPrediction')
        path = os.path.join(rootDir,folder,frames[i+FRA])
        img = Image.open(path)
        frame10 = process_im(np.array(img))/255.
        img_back = frame10

        for j in range(FRA,N_FRAME):

            imgname = os.path.join(rootDir,folder,frames[i+j+1])
            img = Image.open(imgname)
            original = process_im(np.array(img))/255.
        
            tensorinput = torch.from_numpy(img_back).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0)
            ofinput = ofprediction[:, j, :].data.resize(2,128,160).unsqueeze(0).cpu()

            warpedPrediction = warp(tensorinput,ofinput).squeeze(0).permute(1,2,0).cpu().numpy()
            new_img = np.clip(warpedPrediction, 0, 1.)
            meanserror = np.mean( (new_img - original) ** 2 )
            mse.append(meanserror)
            ssim.append(SSIM(original, new_img))


            if FPRE > 0:
                for n in range(FPRE):
                    '''FALTA CAS FPRE>1
                        A MES ALGO FALLA I NO SE QUE'''
                    # path = os.path.join(rootDir, folder, frames[i + FRA + n + 1])
                    # img = Image.open(path)
                    # frame10 = process_im(np.array(img)) / 255.
                    # img_back = frame10
                    for idx in range(FRA-1):
                        pred = ofprediction[:, FRA, :].data.resize(2,128,160).unsqueeze(0).cpu()
                        ofSampleF[:, idx, :] = warp(ofinputData[:, :, 1 + idx, :, :].data.cpu(),
                                                   pred)\
                                                   .squeeze(0).view(2,128*160)

                    ofSampleF[:, FRA-1, :] = warp(ofprediction[:, FRA, :].data.resize(2,128,160).unsqueeze(0).cpu(),
                                                   ofprediction[:, FRA, :]    .data.resize(2,128,160).unsqueeze(0).cpu())\
                                                   .squeeze(0).view(2,128*160)
                    showOFs(ofSampleF[:,0:FRA,:].view(2, FRA, 128, 160).unsqueeze(0).cpu(),
                            ofprediction[:,1:,:].view(2, FRA, 128, 160).unsqueeze(0).cpu(), FRA, 'futurePred1')

                    with torch.no_grad():
                        ofpredictionf = ofmodel.forward(Variable(ofSampleF[:,0:FRA,:]).cuda(gpu_id))
                    ofSampleF[:, FRA, :] = ofpredictionf[:, FRA, :]


                    imgname = os.path.join(rootDir, folder, frames[i + j + n + 2])
                    img = Image.open(imgname)
                    originalf = process_im(np.array(img)) / 255.
                    tensorinputF = torch.from_numpy(new_img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0)
                    ofinputF = ofpredictionf[:, FRA, :].data.resize(2, 128, 160).unsqueeze(0).cpu()

                    showOFs(ofSample[:, FRA+PRE, :].view(2, 128, 160).unsqueeze(0).unsqueeze(2),
                            ofpredictionf[:, FRA, :].view(2, 128, 160).unsqueeze(0).unsqueeze(2), 1, 'futurePred1')
                    print(FRA-j)
                    warpedPrediction = warp(tensorinputF, ofinputF).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img_back_f = np.clip(warpedPrediction, 0, 1.)
                    meanserror = np.mean((img_back_f - originalf) ** 2)
                    msef.append(meanserror)
                    ssimf.append(SSIM(originalf, img_back_f))
                    # msef[n] = meanserror
                    # ssimf[n] = (SSIM(original, img_back_f))

        mse = np.asarray(mse)
        ssim = np.asarray(ssim)
        msef = np.asarray(msef)
        ssimf = np.asarray(ssimf)

        print('mse:',mse,mse.shape)
        print('ssim:',ssim)
        print('msef:',msef,msef.shape)
        print('ssimf:',ssimf)

        fomse.append(mse[:,np.newaxis])
        fossim.append(ssim[:,np.newaxis])
        fomsef.append(msef[:,np.newaxis])
        fossimf.append(ssimf[:,np.newaxis])

fomse = np.concatenate((fomse),1)
fossim = np.concatenate((fossim),1)
fomsef = np.concatenate((fomsef),1)
fossimf = np.concatenate((fossimf),1)

print('fomse shape',fomse.shape)
#print("MSE : ", np.mean(np.array(mse)))
#print("SSIMs : ", np.mean(np.array(ssim)))
print("MSE : ", np.mean(fomse,axis=1))
print("SSIMs : ", np.mean(fossim,axis=1))
print("MSEF : ", np.mean(fomsef,axis=1))
print("SSIMFs : ", np.mean(fossimf,axis=1))

############################################################################
