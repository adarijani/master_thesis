#######################################################
#################### Wirtinger Flow ###################
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
n = 64
m = 10 * n
cplx_flag = 1
T = 800
npower_iter = 30
tau0 = 330


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


def wf(Amatrix, x, y):
    Relerrs = np.zeros((T + 1, 1))
    z0 = torch.randn((n, n2), dtype=torch.cdouble)
    z0 = z0 / torch.linalg.norm(z0)
    for tt in range(npower_iter):
        z0 = Ah(Amatrix, torch.multiply(y, (A(Amatrix, z0))))
        z0 = z0 / torch.linalg.norm(z0)
    normest = math.sqrt(torch.sum(y)) / m
    z0 = normest * z0
    z = z0
    ang_tr = torch.angle(torch.trace(x.H * z))
    diff = x - torch.exp(-1j * ang_tr) * z
    norm = torch.linalg.norm(x)
    Relerrs[0] = torch.linalg.norm(diff) / norm
    normest = math.sqrt(torch.sum(y**2) / m)
    for tt in range(T + 1):
        yz = Amatrix @ z
        mul_diff = torch.multiply(abs(yz) ** 2 - y**2, yz)
        grad = 1 / m * Ah(Amatrix, mul_diff)
        tau = min(1 - math.exp(-tt / tau0), 0.2)
        z = z - tau / normest**2 * grad
        ang_tr = torch.angle(torch.trace(x.H * z))
        diff = x - torch.exp(-1j * ang_tr) * z
        norm = torch.linalg.norm(x)
        Relerrs[tt] = torch.linalg.norm(diff) / norm
    return z, Relerrs


Amatrix, x, y = generator()
z, err = wf(Amatrix, x, y)
print(err, "\n", err[-1])
