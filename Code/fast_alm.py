import numpy as np
from scipy.sparse import csc_matrix, dia_matrix, lil_matrix
import numpy.matlib as npm
import torch
import math
import time

def form_diag_StS(nr, nc, dim):
    N = int(nr / dim + nc - 1)
    S = lil_matrix((nr * nc, N * dim))
    s1 = np.identity(nr, dtype=None)
    s2 = np.zeros((nr, N * dim - nr))
    s = np.concatenate((s1, s2), axis=1)
    for c in range(0, nc):
        # S[c*nr:(c+1)*nr, :] = np.roll(s, (0, c*dim))
        S[c * nr:(c + 1) * nr, :] = np.roll(s, (0, c * dim))
    StS = S.T * S
    diag_StS = take_diag_from_sparse(StS)
    S_tensor = convert_to_sparse_tensor(S)
    StS_tensor = convert_to_sparse_tensor(StS)
    diag_StS_tensor = convert_to_sparse_tensor(diag_StS)

    return diag_StS_tensor, S_tensor, StS_tensor

def form_diag_PtP(Omega, dim, N):
    P = lil_matrix((N * dim, N * dim))
    rmatO = npm.repmat(Omega, dim, 1)
    # reshOmega = np.reshape(rmatO.T, (1, N*dim))
    reshOmega = np.reshape(rmatO.T, (1, -1))  # reshape column-wised
    for i in range(0, N * dim):
        P[i, i] = reshOmega[:, i]
    PtP = P.T * P
    diag_PtP = take_diag_from_sparse(PtP)
    P_tensor = convert_to_sparse_tensor(P)
    PtP_tensor = convert_to_sparse_tensor(PtP)
    diag_PtP_tensor = convert_to_sparse_tensor(diag_PtP)

    return diag_PtP_tensor, PtP_tensor, P_tensor

def convert_to_sparse_tensor(sparseMat):
    coo = sparseMat.tocoo()
    values = coo.data
    indx = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indx)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sparse_Tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()
    return sparse_Tensor

def take_diag_from_sparse(matTmat):
    N = matTmat.shape[0]
    diag_mat = lil_matrix((N, 1))
    for i in range(0, N):
        diag_mat[i, :] = matTmat[i, i]
    return diag_mat

def fastshrink(D, th):
    u, s, v = torch.svd(D)

    s = s + 1e-10
    eigVal = torch.diag(s - th)

    eigVal_new = eigVal.clone()
    eigVal_new[eigVal <= 0] = 0

    A = torch.mm(torch.mm(u, eigVal_new), v.t()).cuda()

    return A

def l2_fast_alm(sequence, Lambda, Omega):

    dim = sequence.shape[0]  # dim of the sequence
    N = sequence.shape[1]  # number of frames
    # Omega = torch.ones((1, N))
    nr = int(math.ceil(N/(dim+1))) * dim    #
    nc = N - int(math.ceil(N/(dim+1)))+1
    Jsize = np.array([nr, nc])
    MaxIter = 1000  #1000
    Tolerence = 1e-7


    vec_u = torch.reshape(sequence.transpose(1, 0), (-1, 1)).cuda()  # 1D tensor
    diag_StS, S, StS = form_diag_StS(nr, nc, dim)   # sparse tensor
    diag_PtP, P, PtP = form_diag_PtP(Omega, dim, N)

    # initialization
    mu = 0.05/(max(vec_u)/10)  # make to a scaler
    rho = torch.FloatTensor([1.05]).cuda()
    h = 1.1*vec_u
    # y = np.zeros((nr*nc,1)) # (4488,1)
    # R = np.zeros(Jsize)
    y = torch.zeros((nr*nc, 1)).cuda()
    R = torch.zeros([nr, nc]).cuda()
    # R = torch.reshape(R.transpose(1, 0), (-1, 1))
    for iter in range(0, MaxIter):
        R = torch.reshape(torch.mm(S, h) + y/mu, (nc, nr)) + 1e-10  # column-wised reshape
        R = R.transpose(1, 0).cuda()
        J = fastshrink(R, 1/mu)
        j = torch.reshape(J.t(), (-1, 1)).cuda()  # column-wised reshape

           # update h
        numerate = Lambda*torch.mm(PtP, vec_u) + torch.mm(mu * S.transpose(1, 0).to_dense(), j - (y / mu))
        denominator = Lambda * diag_PtP.to_dense() + mu * diag_StS.to_dense()
        h = torch.div(numerate, denominator).cuda()
        y = y + mu*(torch.mm(S, h)-j)
        # increase the mu to force the constraint
        mu = mu * rho

        if torch.norm(torch.mm(S, h)-j) < Tolerence:
            break

    # u_estimate = torch.reshape(h, (dim, N))
    u_estimate = torch.reshape(h, (N, dim)).transpose(1, 0).cuda()

    return u_estimate

# if __name__ == "__main__":
#     Lambda = 1
#     # sequence = np.random.rand(2, 30)
#     sequence = torch.ones((2, 18))*2
#     # Omega = torch.ones((1, 9))  # the same size of sequence, containing outliers
#     Omega = torch.Tensor(np.array([[0,1,1,1,0,1,1,0,1],[0,1,1,1,0,1,1,0,1]]))
#     # sequence_out = np.copy(sequence)
#     idex = [0, 4, 7]
#     for id in idex:
#         sequence[:, id] = 0
#     #     Omega[:, id] = 0
#     # start_time = time.clock()
#     # estimate = dataInterpolation.l2_fast_alm(sequence, Lambda, Omega)
#     estimate = l2_fast_alm(sequence, Lambda, Omega)
#     # print(time.clock() - start_time, "seconds")
#     print(estimate)
#     # # diff_reout = abs(estimate-sequence)
#     # # diff_orig = abs(sequence-sequence_out)
#     # # mse_reout = sum(sum(np.power(diff_reout, 2)))/60
#     # # mse_orig = sum(sum(np.power(diff_orig, 2)))/60
#     # # print(mse_orig)
#     # # print(mse_reout)
#     #
#     # print('ok')
#
#     # S = form_S(nr=6, nc=6, dim=2)
#     # sequence_out = torch.rand([2, 9])
#     # Omega = torch.ones([1, 9])
#     # Lambda = torch.FloatTensor([1])
#     # estimate = l2_fast_alm(sequence_out, Lambda=Lambda, Omega=Omega)
#     print('ok')