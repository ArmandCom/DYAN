############################# Import Section #################################

## Imports related to PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib as plt
from matplotlib import *
from pylab import *

import time
from math import sqrt
import numpy as np
import random

############################# Import Section #################################

# Create Dictionary
def creatRealDictionary(T, Drr, Dtheta, gpu_id):
    WVar = []
    # Wones = torch.ones(1).cuda(gpu_id)
    # Wones = Variable(Wones, requires_grad=False)

    for i in range(0, T): # matrix 8
        W1 = torch.mul(torch.pow(Drr, i), torch.cos(i * Dtheta))
        W2 = torch.mul(torch.pow(-Drr, i), torch.cos(i * Dtheta))
        W3 = torch.mul(torch.pow(Drr, i), torch.sin(i * Dtheta))
        W4 = torch.mul(torch.pow(-Drr, i), torch.sin(i * Dtheta))
        W = torch.cat((W1, W2, W3, W4), 0)
        WVar.append(W.view(1, -1))
    dic = torch.cat((WVar), 0)
    # G = torch.norm(dic, p=2, dim=0)
    # idx = (G == 0).nonzero()
    # nG = G.clone()
    # nG[idx] = np.sqrt(T)
    # G = nG
    # dic = dic / G

    mean_val = torch.mean(dic, 0).unsqueeze(0).data
    dic = (dic - mean_val)
    std = dic.std(0).unsqueeze(0).data
    std[std == 0] = 1
    dic = dic / std

    return dic

def fista(D, Y, lam, maxIter, gpu_id):
    DtD = torch.matmul(torch.t(D), D)
    L = torch.norm(DtD, 2)
    linv = 1 / L
    DtY = torch.matmul(torch.t(D), Y)
    x_old = Variable(torch.zeros(DtD.shape[1], DtY.shape[2]).cuda(gpu_id), requires_grad=True)
    t = 1
    y_old = x_old
    lambd = lam * (linv.data.cpu().numpy())
    A = Variable(torch.eye(DtD.shape[1]).cuda(gpu_id), requires_grad=True) - torch.mul(DtD, linv)


    DtY = torch.mul(DtY, linv)

    Softshrink = nn.Softshrink(lambd)
    with torch.no_grad():

        for ii in range(maxIter):
            Ay = torch.matmul(A, y_old)
            del y_old
            with torch.enable_grad():

                x_new = Softshrink((Ay + DtY))

            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
            tt = (t - 1) / t_new
            y_old = torch.mul(x_new, (1 + tt))
            y_old -= torch.mul(x_old, tt)
            if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-4:
                x_old = x_new
                break
            t = t_new
            x_old = x_new

            del x_new
    # print('c sum: ', torch.sum(x_old))
    return x_old


class Encoder(nn.Module):
    def __init__(self, Drr, Dtheta, T, lam, gpu_id):
        super(Encoder, self).__init__()

        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.T = T
        self.gid = gpu_id
        self.lam = lam

    def forward(self, x):
        # polar plot
        """
        if random.randint(1,20) == 1:
            rr = self.rr.data.cpu().detach().numpy()
            theta = self.theta.data.cpu().detach().numpy()
            ax = plt.subplot(1, 1, 1, projection='polar')
            ax.scatter(0, 1, c='black')
            # unactive poles
            ax.scatter(theta, rr)
            ax.scatter(-theta, rr)
            ax.scatter(np.pi - theta, rr)
            ax.scatter(theta - np.pi, rr)
            #
            ax.set_rmax(1.2)
            ax.set_title("Dictionary", va='bottom')
            plt.savefig('usedPolesDCT_lam0.png')
            plt.show()
            plt.close()"""
        dic = creatRealDictionary(self.T, self.rr, self.theta, self.gid)
        sparsecode = fista(dic, x, self.lam, 100, self.gid)

        '''
        plt.figure()
        plt.pcolormesh(torch.t(sparsecode[0,:,:]).data.cpu().detach().numpy(), cmap='RdBu')
        plt.colorbar()
        plt.savefig('C_evolution_subsIni.png') #, dpi=200
        # plt.show()
        plt.close()'''

        return Variable(sparsecode)


class Decoder(nn.Module):
    def __init__(self, rr, theta, T, PRE, gpu_id):
        super(Decoder, self).__init__()

        self.rr = rr
        self.theta = theta
        self.T = T
        self.PRE = PRE
        self.gid = gpu_id

    def forward(self, x):
        dic = creatRealDictionary(self.T + self.PRE, self.rr, self.theta, self.gid)
        result = torch.matmul(dic, x)
        return result


class OFModel(nn.Module):
    def __init__(self, Drr, Dtheta, T, PRE, lam, gpu_id):
        super(OFModel, self).__init__()
        self.l1 = Encoder(Drr, Dtheta, T, lam, gpu_id)
        self.l2 = Decoder(self.l1.rr, self.l1.theta, T, PRE, gpu_id)

    def forward(self, x):
        # # ini_val = x[:,0,:].unsqueeze(1)
        # # x = x - ini_val
        # return self.l2(self.l1(x))#+ini_val
        mean_val = torch.mean(x,1).unsqueeze(1)
        x = (x - mean_val)

        std = x.std(1).unsqueeze(1)
        std[std==0] = 1
        x = x/std
        return ((self.l2(self.l1(x)))*std) + mean_val

    def forward2(self, x):
        return self.l1(x)
