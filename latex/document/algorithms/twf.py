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
grad_type = "TWF_Poiss"
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
def twf(Amatrix, x, y):
    mu = 0.0
    tau0 = 0
    if grad_type == "TWF_Poiss":
        mu = 0.2
    y = y**2
    alpha_y = 3
    alpha_ub = 5
    alpha_lb = 0.3
    alpha_h = 5
    Relerrs = np.zeros((T + 1, 1))
    z0 = torch.randn((n, n2), dtype=torch.cdouble)
    z0 = z0 / torch.linalg.norm(z0)
    normest = math.sqrt(torch.sum(y)) / m
    for tt in range(npower_iter):
        mask = torch.multiply(1, (torch.abs(y) > alpha_y**2 * normest**2))
        ytr = torch.mul(y, mask)
        z0 = Ah(Amatrix, torch.multiply(ytr, (A(Amatrix, z0))))
        z0 = z0 / torch.linalg.norm(z0)
    z0 = normest * z0
    z = z0
    Relerrs[0] = torch.linalg.norm(
        x - torch.exp(-1j * torch.angle(torch.trace(x.H * z))) * z
    ) / torch.linalg.norm(x)
    for tt in range(T + 1):
        yz = Amatrix @ z
        Kt = 1 / m * torch.linalg.norm(torch.abs(yz) ** 2 - y, 1)
        if grad_type == "TWF_Poiss":
            temp_ub = torch.abs(yz) / torch.linalg.norm(z)
            Eub = torch.multiply(1, (torch.abs(temp_ub) <= alpha_ub))
            temp_lb = torch.abs(yz) / torch.linalg.norm(z)
            Elb = torch.multiply(1, (torch.abs(temp_lb) >= alpha_lb))
            temp_h = torch.abs(y - torch.abs(yz) ** 2)
            thre = (alpha_h * Kt / torch.linalg.norm(z)) * torch.linalg.norm(yz)
            Eh = torch.multiply(1, (torch.abs(temp_h) <= thre))
            temp = torch.mul(
                2 * torch.div(torch.abs(yz) ** 2 - y, torch.abs(yz) ** 2), yz
            )
            temp = torch.mul(temp, Eub)
            temp = torch.mul(temp, Elb)
            temp = torch.mul(temp, Eh)
            grad = 1 / m * Ah(Amatrix, temp)
        if grad_type == "TWF_Poiss":
            z = z - mu * grad
        Relerrs[tt] = torch.linalg.norm(
            x - torch.exp(-1j * torch.angle(torch.trace(x.H * z))) * z
        ) / torch.linalg.norm(x)
    return z, Relerrs
Amatrix, x, y = generator()
z, err = twf(Amatrix, x, y)
print(err, "\n", err[-1])
