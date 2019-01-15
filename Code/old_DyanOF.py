############################# Import Section #################################

## Imports related to PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

from math import sqrt
import numpy as np


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

# Seems that FISTA is the algorithm to maximize sparsity of the representation of the inputs through the Dict. (review)
def fista(D, Y, lam, maxIter, gpu_id):
    DtD = torch.matmul(torch.t(D), D)  # product of tensor t(D)-->D' and D --> D = dictionary
    L = torch.norm(DtD, 2)
    linv = 1 / L # inverse of the norm
    DtY = torch.matmul(torch.t(D), Y)
    x_old = Variable(torch.zeros(DtD.shape[1], DtY.shape[2]).cuda(gpu_id), requires_grad=True)
    t = 1 #initial state
    y_old = x_old  # inicialize x and y at 0
    lambd = lam * (linv.data.cpu().numpy())
    A = Variable(torch.eye(DtD.shape[1]).cuda(gpu_id), requires_grad=True) - torch.mul(DtD, linv)
    # A = I(_eye_) - 1/L(DtD)

    DtY = torch.mul(DtY, linv)
    # b = DtY - lambd?

    Softshrink = nn.Softshrink(lambd)  # applies shrinkage function elementwise - act. function
    with torch.no_grad():

        for ii in range(maxIter):
            Ay = torch.matmul(A, y_old) #y = gamma_t
            del y_old
            with torch.enable_grad():
                x_new = Softshrink((Ay + DtY)) #x = c_t
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2. #t = s_t
            tt = (t - 1) / t_new # = (s_t - 1)/(s_(t+1))
            y_old = torch.mul(x_new, (1 + tt))
            y_old -= torch.mul(x_old, tt) # gamma_(t+1) = c_(t+1) + (s_t -1)(c_(t+1) - y_t)/s_(t+1) --> y_t = x_old, c_(t+1) = x_new
            if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-4:
                x_old = x_new
                break
            t = t_new
            x_old = x_new
            del x_new
    # print(x_old.shape) # retorna 1x161xn_pixels. Es a dir: un vector sparce code per cada pixel. x_old = c*p o c?
    return x_old


class Encoder(nn.Module):
    def __init__(self, Drr, Dtheta, T, lam, gpu_id): #Drr = D_rho (modul..?), Dtheta = phase, T = Length of dict
        super(Encoder, self).__init__()

        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.T = T
        self.gid = gpu_id
        self.lam = lam

    def forward(self, x):
        dic = creatRealDictionary(self.T, self.rr, self.theta, self.gid)
        sparsecode = fista(dic, x, self.lam, 100, self.gid)

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
        return self.l2(self.l1(x))

    def forward2(self, x):
        return self.l1(x)
