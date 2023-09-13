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
mu = 0.8 + 0.4 * cplx_flag
T = 800
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
def rwf(Amatrix, x, y):
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
        yz = Amatrix @ z
        yz_abs = torch.abs(yz)
        first_divide = torch.divide(yz, yz_abs)
        first_multi = torch.multiply(y, first_divide)
        sub = yz - first_multi
        second_multi = Ah(Amatrix, sub)
        second_divide = torch.divide(second_multi, m)
        z = z - mu * second_divide
        Relerrs[t] = torch.linalg.norm(
            x - torch.exp(-1j * torch.angle(torch.trace(x.H * z))) * z
        ) / torch.linalg.norm(x)
    return z, Relerrs
Amatrix, x, y = generator()
z, err = rwf(Amatrix, x, y)
print(err, "\n", err[-1])
