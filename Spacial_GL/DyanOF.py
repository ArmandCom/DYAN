############################# Import Section #################################

## Imports related to PyTorch
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

from math import sqrt
import time
import numpy as np
from utils import getCells_stride_Kitti


############################# Import Section #################################

# Create Dictionary
def creatRealDictionary(T, Drr, Dtheta, gpu_id):
    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    Wones = Variable(Wones, requires_grad=False)

    for i in range(0, T): # matrix 8
        W1 = torch.mul(torch.pow(Drr, i), torch.cos(i * Dtheta))
        W2 = torch.mul(torch.pow(-Drr, i), torch.cos(i * Dtheta))
        W3 = torch.mul(torch.pow(Drr, i), torch.sin(i * Dtheta))
        W4 = torch.mul(torch.pow(-Drr, i), torch.sin(i * Dtheta))
        W = torch.cat((Wones, W1, W2, W3, W4), 0)
        WVar.append(W.view(1, -1))
    dic = torch.cat((WVar), 0)
    G = torch.norm(dic, p=2, dim=0)
    idx = (G == 0).nonzero()
    nG = G.clone()
    nG[idx] = np.sqrt(T) # T = time horizon?
    G = nG

    dic = dic / G
    # print ('dic_shape: ', dic.shape) #size = 9x161 (161 = 40 simbols/quadrant x4 quadrants, N) (9 o 10 = input o output, finestra temporal (K))
    # conte els pols unicament
    return dic

# def softshrink(x, lambd, gpu_id):
#     t0=time.time()
#     One = Variable(torch.ones(1).cuda(gpu_id))
#     Zero = Variable(torch.zeros(1).cuda(gpu_id))
#     ids = np.arange(128 * 160).reshape(128, 160)  # Kitti dimensions
#     cell_ids = getCells_stride_Kitti(ids, 2)
#
#     lambd_t = One * lambd
#     subgrads = torch.zeros(2,161,128*160).cuda(gpu_id)
#
#     t1 = time.time()
#     #comp = np.zeros(128*160)
#     for cell in cell_ids:
#         #t2 = time.time()
#         print(x.shape)
#         xx_old = torch.norm(x[:,:,cell],2,2)
#         xx_old[xx_old==0]=lambd_t/1000
#         subgrad = torch.max(One - torch.div(lambd_t, xx_old), Zero)
#         print(subgrad.shape)
#         subgrad = subgrad.expand(cell.shape[0], 2, 161).permute(1, 2, 0)
#         idx = 0
#
#         for i in cell:
#             subgrads[:,:,i] = subgrad[:,:,idx]
#             #comp[i] += 1
#             idx += 1
#         #t3 = time.time()
#     x = x*subgrads
#     #print('Times: Declare-',t1-t0,' compute 1 cell-', t3-t2,' 1 softshrink-', time.time()-t0)
#
#
#     return x

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

    # subgrads = torch.zeros(nch,161,128*160).cuda(gpu_id)

    t1 = time.time()
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

def fista(D, Y, lambd, maxIter, gpu_id):
    DtD = torch.matmul(torch.t(D), D) 
    L = torch.norm(DtD, 2)
    linv = 1 / L
    DtY = torch.matmul(torch.t(D), Y)
    x_old = Variable(torch.zeros(DtD.shape[1], DtY.shape[2]).cuda(gpu_id), requires_grad=True)
    t = 1 
    y_old = x_old  # inicialize x and y at 0
    lambd = lambd * (linv.data.cpu().numpy())
    A = Variable(torch.eye(DtD.shape[1]).cuda(gpu_id), requires_grad=True) - torch.mul(DtD, linv)

    DtY = torch.mul(DtY, linv)
    Softshrink = nn.Softshrink(lambd)
    with torch.no_grad():

        for ii in range(maxIter):
            Ay = torch.matmul(A, y_old) #y = gamma_t
            del y_old
            with torch.enable_grad():
                x_new = Softshrink((Ay + DtY)) #,lambd,gpu_id
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
    return x_old


class Encoder(nn.Module):
    def __init__(self, Drr, Dtheta, T, gpu_id): #Drr = D_rho (modul..?), Dtheta = phase, T = Length of dict
        super(Encoder, self).__init__()

        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.T = T
        self.gid = gpu_id

    def forward(self, x):
        dic = creatRealDictionary(self.T, self.rr, self.theta, self.gid)
        sparsecode = fista(dic, x, 0.1, 100, self.gid)

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
    def __init__(self, Drr, Dtheta, T, PRE, gpu_id):
        super(OFModel, self).__init__()
        self.l1 = Encoder(Drr, Dtheta, T, gpu_id)
        self.l2 = Decoder(self.l1.rr, self.l1.theta, T, PRE, gpu_id)

    def forward(self, x):
        return self.l2(self.l1(x))

    def forward2(self, x):
        return self.l1(x)
