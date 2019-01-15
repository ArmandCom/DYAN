############################# Import Section #################################

## Imports related to PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import numpy as np

############################# Import Section #################################


# Create Dictionary
def creatRealDictionary(T, Drr, Dtheta, gpu_id):
    WVar = []
    gpu_id = gpu_id
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
    nG[idx] = np.sqrt(T)
    G = nG

    dic = dic / G
    return dic

def fista(D, Y, lambd, maxIter):
    DtD = torch.matmul(torch.t(D), D)
    L = torch.norm(DtD, 2)
    linv = 1 / L
    DtY = torch.matmul(torch.t(D), Y)
    x_old = Variable(torch.zeros(DtD.shape[1], DtY.shape[2]).cuda(Y.get_device()), requires_grad=True)
    t = 1
    y_old = x_old
    lambd = lambd * (linv.data.cpu().numpy())
    A = Variable(torch.eye(DtD.shape[1]).cuda(Y.get_device()), requires_grad=True) - torch.mul(DtD, linv)


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
    return x_old


class Encoder(nn.Module):
    def __init__(self, Drr, Dtheta, T):
        super(Encoder, self).__init__()

        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.T = T

    def forward(self, x):
        dic = creatRealDictionary(self.T, self.rr, self.theta, x.get_device())
        sparsecode = fista(dic, x, 0.05, 100)

        return Variable(sparsecode)


class Decoder(nn.Module):
    def __init__(self, rr, theta, T, PRE):
        super(Decoder, self).__init__()

        self.rr = rr
        self.theta = theta
        self.T = T
        self.PRE = PRE

    def forward(self, x):
        dic = creatRealDictionary(self.T + self.PRE, self.rr, self.theta, x.get_device())
        result = torch.matmul(dic, x)
        return result

class OFModel(nn.Module):
    def __init__(self, DrrT, DthetaT, DrrSH, DthetaSH, DrrSV, DthetaSV, T, PRE, x_fra, y_fra):
        super(OFModel, self).__init__()

        self.nch = 4
        # to start, we divide each frame in four chunks, vertically and horizontally
        self.Wx = int(x_fra/self.nch) # Window size at rows
        self.Strx = int(x_fra/self.nch) # Stride at rows
        self.Wy = int(y_fra/self.nch)
        self.Stry = int(y_fra/self.nch)

        self.Clen = len(DrrT)*4 + 1
        self.T = T
        self.x_fra = x_fra
        self.y_fra = y_fra
        self.nChunks = int(self.x_fra/self.Wx)

        self.et = Encoder(DrrT, DthetaT, T)
        self.esH = Encoder(DrrSH, DthetaSH, self.Wx)
        self.esV = Encoder(DrrSV, DthetaSV, self.Wy)

        self.dt = Decoder(self.et.rr, self.et.theta, T, PRE)
        self.dsH = Decoder(self.esH.rr, self.esH.theta, self.Wx, 0)
        self.dsV = Decoder(self.esH.rr, self.esH.theta, self.Wy, 0)


    # Mirar que no comparteixin diccionari temporal i espaials
    # def forward(self, x):
    #     # TODO Multiprocessing with multiprocessing.set_start_method('spawn') to split model in gpu's
    #     # TODO Block DataParallel
    #
    #     gpu_id = x.get_device()
    #     out = torch.empty(2*self.nChunks, self.T, (self.Wx + self.Wy) * self.Clen).cuda(gpu_id)
    #     for i in range(self.T):
    #         inputFrame = Variable(x[:, i, :].view(-1, self.x_fra, self.y_fra))
    #
    #         inpChunkH = torch.cat(list(inputFrame.unfold(1, self.Wx, self.Strx)), dim=0)
    #         inpChunkV = torch.cat(list(inputFrame.permute(0, 2, 1).unfold(1, self.Wy, self.Stry)), dim=0)
    #
    #         outputH, outputV = [self.esH(inpChunkH), self.esV(inpChunkV)] # We probably would be faster if we used 1 encoder for all
    #
    #         output = torch.cat((outputH.view(-1, self.Clen * self.Wx), outputV.view(-1, self.Clen * self.Wy)), 1)
    #         # Are output values correct? 60 is a big number for a c --> they seem to vary a lot
    #         out[:, i, :] = output
    #
    #     cPred = self.dt(self.et(out))
    #     cPredH = cPred[:, self.T, 0:(self.Clen * self.Wx)].view(2*self.nChunks, self.Clen, self.Wx)
    #     cPredV = cPred[:, self.T, (self.Clen * self.Wx): ].view(2*self.nChunks, self.Clen, self.Wy)
    #
    #     outH, outV = [self.dsH(cPredH), self.dsV(cPredV)]
    #
    #     outH, outV = [outH.permute(0, 2, 1).contiguous().view(2, self.x_fra, self.y_fra),
    #                   outV.permute(0, 2, 1).contiguous().view(2, self.y_fra, self.x_fra)]
    #
    #     # pred = ((outH + outV.permute(0, 2, 1)) / 2).view(-1, self.x_fra * self.y_fra)
    #
    #     return outH, outV.permute(0,2,1)

    def forward(self, x):

        gpu_id = x.get_device()
        out = torch.empty(2*self.nChunks, self.T, (self.x_fra + self.y_fra) * self.Clen).cuda(gpu_id)

        for i in range(self.T):
            inputFrame = Variable(x[:, i, :].view(-1, self.x_fra, self.y_fra))

            # Unfold input
            inpChunkH = torch.cat(list(inputFrame
                                       .unfold(1, self.Wx, self.Strx)), dim=0).permute(0,2,1)
            inpChunkV = torch.cat(list(inputFrame.permute(0, 2, 1)
                                       .unfold(1, self.Wy, self.Stry)), dim=0).permute(0,2,1)

            # Unfold inverted input
            # inpChunkHI = torch.cat(list(inputFrame[:, :, np.linspace(self.y_fra, 1, num=self.y_fra)-1]
            #                             .unfold(1, self.Wx, self.Strx)), dim=0)
            # inpChunkVI = torch.cat(list(inputFrame[:, :, np.linspace(self.x_fra, 1, num=self.x_fra)-1].permute(0, 2, 1)
            #                             .unfold(1, self.Wy, self.Stry)), dim=0)

            outputH, outputV = [self.esH(inpChunkH), self.esV(inpChunkV)]

            output = torch.cat((outputH.view(-1, self.Clen * self.y_fra), outputV.view(-1, self.Clen * self.x_fra)), 1)

            out[:, i, :] = output

        c_arrayH = out[:, 0:9, 0:(self.Clen * self.y_fra)].view(2 * self.nChunks, 9, self.Clen, self.y_fra)

        cPred = self.dt(self.et(out))

        # c_arrayH = cPred[:, 0:10, 0:(self.Clen * self.y_fra)].view(2 * self.nChunks, 10, self.Clen, self.y_fra)
        px_id = 80
        c_ev = c_arrayH[3,:,:,px_id].data
        plt.pcolormesh(torch.t(c_ev), cmap='RdBu')
        # plt.colorbar()
        plt.savefig('C_evolution_col%d_ch%d_TFrom%dTo%d_%s' % (px_id, 3, 0, 9, 'Vert'))
        plt.close()


        cPredH_rec = cPred[:, self.T -1, 0:(self.Clen * self.y_fra)].view(2*self.nChunks, self.Clen, self.y_fra)
        cPredV_rec = cPred[:, self.T -1, (self.Clen * self.y_fra):].view(2 * self.nChunks, self.Clen, self.x_fra)
        cPredH = cPred[:, self.T, 0:(self.Clen * self.y_fra)].view(2*self.nChunks, self.Clen, self.y_fra)
        cPredV = cPred[:, self.T, (self.Clen * self.y_fra): ].view(2*self.nChunks, self.Clen, self.x_fra)

        outH, outV, outH_rec, outV_rec = [self.dsH(cPredH), self.dsV(cPredV), self.dsH(cPredH_rec), self.dsV(cPredV_rec)]

        outH, outV, outH_rec, outV_rec= [outH.contiguous().view(2, self.x_fra, self.y_fra),
                      outV.contiguous().view(2, self.y_fra, self.x_fra), outH_rec.contiguous().view(2, self.x_fra, self.y_fra),
                      outV_rec.contiguous().view(2, self.y_fra, self.x_fra)]

        # pred = ((outH + outV.permute(0, 2, 1)) / 2).view(-1, self.x_fra * self.y_fra)

        return outH, outV.permute(0,2,1), outH_rec, outV_rec.permute(0,2,1)

    def forwardE(self, x):
        gpu_id = x.get_device()
        out = torch.empty(2 * self.nChunks, self.T, (self.x_fra + self.y_fra) * self.Clen).cuda(gpu_id)

        for i in range(self.T):
            inputFrame = Variable(x[:, i, :].view(-1, self.x_fra, self.y_fra))

            # Unfold input
            inpChunkH = torch.cat(list(inputFrame
                                       .unfold(1, self.Wx, self.Strx)), dim=0).permute(0, 2, 1)
            inpChunkV = torch.cat(list(inputFrame.permute(0, 2, 1)
                                       .unfold(1, self.Wy, self.Stry)), dim=0).permute(0, 2, 1)

            outputH, outputV = [self.esH(inpChunkH), self.esV(inpChunkV)]

            output = torch.cat((outputH.view(-1, self.Clen * self.y_fra), outputV.view(-1, self.Clen * self.x_fra)), 1)

            out[:, i, :] = output

        c = self.et(out)
        return c

    def forwardD(self, c):

        cPred = self.dt(c)

        cPredH = cPred[:, self.T, 0:(self.Clen * self.y_fra)].view(2 * self.nChunks, self.Clen, self.y_fra)
        cPredV = cPred[:, self.T,  (self.Clen * self.y_fra):].view(2 * self.nChunks, self.Clen, self.x_fra)

        outH, outV = [self.dsH(cPredH), self.dsV(cPredV)]

        outH, outV = [outH.contiguous().view(2, self.x_fra, self.y_fra),
                      outV.contiguous().view(2, self.y_fra, self.x_fra)]

        return outH, outV.permute(0, 2, 1)
