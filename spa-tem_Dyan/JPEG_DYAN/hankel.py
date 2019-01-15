import numpy as np
from scipy.sparse import csc_matrix
import numpy.matlib as npm
import torch

def form_block_hankel(y,Hsize):
    dim, N = y.shape  # y = (1,28)
    if dim > N:
        print('make it to the row vector')
    nr = Hsize[0]
    nc = Hsize[1]
    # H = np.zeros((nr, nc))  # 2x(N-m+1), eg: 2x(28-2+1) = 2 x 27
    H = torch.zeros((nr, nc))
    # ridex = np.expand_dims(np.linspace(0, nr-1, nr, dtype=np.int), axis=1)  # col vector
    # cidex = np.expand_dims(np.linspace(0, nc-1, nc, dtype=np.int), axis=0)  # row vector
    ridex = torch.linspace(0, nr-1, nr).unsqueeze(1)
    cidex = torch.linspace(0, nc-1, nc).unsqueeze(0)

    # p1 = ridex*np.ones((nr,nc), dtype=np.int)

    # p2 = np.repeat(cidex, nr, axis=0)
    p1 = ridex*torch.ones([nr, nc])
    p2 = cidex.expand(nr, nc)
    # H = np.zeros((nr,nc), dtype=np.int)
    # Hidex = p1 + p2
    Hidex = p1.add(p2)  # convert datatype to longTensor
    reHidex = torch.reshape(Hidex, (1, -1))
    reHidex = reHidex.type(torch.long)
    # yh = y[:,reHidex]
    # H = yh.reshape(nr, nc)
    H = torch.reshape(y[:, reHidex], (nr, nc)).cuda()

    # t = y.T

    # for r in range(0, nr):
    #     for c in range(0, nc):
    #         # idx = int(Hidex[r, c])
    #         idx = int(Hidex[r, c])
    #         H[r, c] = y[:, idx]
    return H



if __name__ == "__main__":
    # y = torch.randn([1, 28])
    y = torch.arange(27).float().unsqueeze(0)
    # y = np.ones([2,28])
    print(y)
    Hsize = torch.Size([14, 14])
    H = form_block_hankel(y, Hsize)

    print(H)
    print('ok')
