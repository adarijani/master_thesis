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


def irwf(Amatrix, x, y):
    Relerrs = np.zeros((T + 1, 1))
    z0 = torch.randn((n, n2), dtype=torch.cdouble)
    torch.linalg.norm(z0)
    normest = (
        (math.sqrt(math.pi / 2) * (1 - cplx_flag) + math.sqrt(4 / math.pi) * cplx_flag)
        * torch.sum(y)
        / m
    )
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
        yz_batch = A(Asub, z)
        y_batch = y[shuffled_indices]
        yz_abs_batch = torch.abs(yz_batch)
        first_divide_batch = torch.divide(yz_batch, yz_abs_batch)
        first_multi_batch = torch.multiply(y_batch, first_divide_batch)
        sub_batch = yz_batch - first_multi_batch
        second_multi_batch = Ah(Asub, sub_batch)
        second_divide_batch = torch.divide(second_multi_batch, m)
        z = z - mu * second_divide_batch
        Relerrs[t] = torch.linalg.norm(
            x - torch.exp(-1j * torch.angle(torch.trace(x.H * z))) * z
        ) / torch.linalg.norm(x)
    return z, Relerrs


Amatrix, x, y = generator()
z, err = irwf(Amatrix, x, y)
print(err, "\n", err[-1])
