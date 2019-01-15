############################# Import Section #################################

## Imports related to PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import numpy as np
from utilsStd import creatRealDictionary

############################# Import Section #################################

gpu_id = 1

def softshrink(x, lambd, gpu_id):
    nch = 2
    nx = 161
    # xs = 128
    # ys = 160
    ws = 5

    t0=time.time()
    One = Variable(torch.ones(1).cuda(gpu_id), requires_grad=False)
    Zero = Variable(torch.zeros(1).cuda(gpu_id), requires_grad=False)
    lambd_t = One * lambd

    x = x.view(nch,nx,128,160)
    poolL2 = nn.LPPool2d(2,ws,stride=ws, ceil_mode=True)
    xx_old = poolL2(x)
    xx_old[xx_old == 0] = lambd_t / 1000
    subgrad = torch.max(One - torch.div(lambd_t, xx_old), Zero)
    xs = subgrad.shape[2]
    ys = subgrad.shape[3]
    subgrad = subgrad.view(nch,nx,xs*ys,-1).repeat(1,1,1,ws).view(nch,nx,-1,ws*ys).repeat(1,1,1,ws).view(nch,nx,-1,ws*ys)[:,:,:128,:]
    x = (x*subgrad).view(nch,nx,-1,20480).squeeze()
    # print('total time per subgrad op: ', time.time()-t0)
    # print('done!')

    return x

def fista(D, Y, Gamma,maxIter):

    DtD = torch.matmul(torch.t(D),D)
    # wen: only 2norm be the same as max(eig(DtD))! checked with Matlab
    L = torch.norm(DtD,p=2)
    if L == 0:
        print("Got L == 0")
        linv = 0
    else:
        linv = 1/L
    DtY = torch.matmul(torch.t(D),Y)
    #scipy.io.savemat('DtD.mat', mdict={'DtD': DtD})
    x_old = Variable(torch.zeros(DtD.shape[1],DtY.shape[2]).cuda(gpu_id), requires_grad=True)
    t = 1
    y_old = x_old
    Gamma = Gamma*linv
    A = Variable(torch.eye(DtD.shape[1]).cuda(gpu_id),requires_grad=True) - torch.mul(DtD,linv)

    DtY = torch.mul(DtY,linv)

    lambd = Gamma.view(-1,1).expand(-1,DtY.shape[2]) #
    for ii in range(maxIter):
        Ay = torch.matmul(A,y_old)
        del y_old
        x_new = Ay +DtY# - lambd
        x_new = torch.sign(x_new)*torch.max(torch.abs(x_new)- lambd,torch.zeros_like(x_new))
        # x_new = softshrink(x_new, lambd,gpu_id)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt)
        t = t_new
        x_old = x_new
        del x_new

    return x_old



class SClayer(nn.Module):
    def __init__(self, Drr, Dtheta, Gamma, T):
        super(SClayer, self).__init__()

        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.gamma = nn.Parameter(Gamma)
        self.T = T

    def forward(self, x):
        dic = creatRealDictionary(self.T,self.rr,self.theta)
        sparsecode = fista(dic,x,0.05,40)
        return Variable(sparsecode)

class invSClayer(nn.Module):
    def __init__(self,rr,theta, T, PRE):
        super(invSClayer,self).__init__()

        self.rr = rr
        self.theta = theta
        self.T = T
        self.PRE = PRE

    def forward(self,x):
        dic = creatRealDictionary(self.T+self.PRE,self.rr,self.theta)
        result = torch.matmul(dic,x)
        return result


class SC2(nn.Module):
    def __init__(self, Drr, Dtheta, Gamma, T, PRE):
        super(SC2, self).__init__()
        self.l1 = SClayer(Drr, Dtheta, Gamma, T)
        self.l2 = invSClayer(self.l1.rr,self.l1.theta, T, PRE)

    def forward(self,x):
        mean_val = torch.mean(x,1).unsqueeze(1)
        x = (x - mean_val)

        std = x.std(1).unsqueeze(1)
        std[std==0] = 1
        x = x/std
        return ((self.l2(self.l1(x)))*std) + mean_val

    def forward2(self,x):
        return self.l1(x)

