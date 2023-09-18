#######################################################
### Incrementally Minibatch Reshaped Wirtinger Flow ###
#######################################################
import torch
import numpy as np
import math

cuda_opt = True
if torch.cuda.is_available() & cuda_opt:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
n2 = 1
n = 640
alpha = 10
m = alpha * n
cplx_flag = 1
mu = 0.8 + 0.4 * cplx_flag
# mu = 1.2
T = 200
# T = 800
# batch_factor = 5
# batch = int(batch_factor*n)
batch = m
npower_iter = 30


def A(Amatrix, X):
    return torch.linalg.matmul(Amatrix, X)


def Ah(Amatrix, Z):
    return torch.linalg.matmul(Amatrix.adjoint(), Z)


def generator():
    x = (
        torch.randn((n, 1), dtype=torch.double)
        + torch.randn((n, 1), dtype=torch.double) * 1j
    )
    Amatrix = (
        torch.randn((m, n), dtype=torch.double)
        + torch.randn((m, n), dtype=torch.double) * 1j
    ) / math.sqrt(2) ** cplx_flag
    y = torch.abs(A(Amatrix, x))
    return Amatrix, x, y


def imrwf(Amatrix, x, y):
    Relerrs = np.zeros((T + 1, 1))
    z0 = torch.randn((n, n2), dtype=torch.cdouble)
    torch.linalg.norm(z0)
    n_r = math.sqrt(math.pi / 2) * (1 - cplx_flag)
    n_c = math.sqrt(4 / math.pi) * cplx_flag
    n_y = torch.sum(y) / m
    normest = (n_r + n_c) * n_y
    ytr = torch.multiply(y, (torch.abs(y) > 1 * normest))
    for tt in range(npower_iter):
        z0 = Ah(Amatrix, torch.multiply(ytr, (A(Amatrix, z0))))
        z0 = z0 / torch.linalg.norm(z0)
    z0 = normest * z0
    z = z0
    Relerrs[0] = torch.linalg.norm(
        x - torch.exp(-1j * torch.angle(torch.trace(x.H * z))) * z
    ) / torch.linalg.norm(x)
    for t in range(T + 1):
        shuffled_indices = torch.randperm(batch)
        Asub = Amatrix[shuffled_indices, :]
        yz_b = A(Asub, z)
        y_b = y[shuffled_indices]
        yz_abs_b = torch.abs(yz_b)
        first_div_b = torch.divide(yz_b, yz_abs_b)
        first_mul_b = torch.multiply(y_b, first_div_b)
        sub_b = yz_b - first_mul_b
        second_multi_b = Ah(Asub, sub_b)
        second_divide_b = torch.divide(second_multi_b, m)
        z = z - mu * second_divide_b
        ang_tr = torch.angle(torch.trace(x.H * z))
        diff = x - torch.exp(-1j * ang_tr) * z
        norm = torch.linalg.norm(x)
        Relerrs[t] = torch.linalg.norm(diff) / norm
    return z, Relerrs


Amatrix, x, y = generator()
z, err = imrwf(Amatrix, x, y)
print(err, "\n", err[-1])
