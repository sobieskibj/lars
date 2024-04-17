import torch
from torch import nn
import torch.nn.functional as F

class LARS(nn.Module):

    def __init__(self, p: int):
        super().__init__()
        self.p = p
        self.beta = torch.zeros(self.p, 1)

    def fit(self, dataset):
        X, y = dataset.get_data()

        mu_hat = torch.zeros_like(y)
        s = torch.zeros_like(X[0, None].T)

        for k in range(self.p):

            c_hat = X.T @ (y - mu_hat)

            C_hat = c_hat.abs().max()

            # probably not needed as we should start with empty A
            A = c_hat.abs() == C_hat
            
            s[A] = c_hat[A].sign()

            X_A = X @ s
            G_A = X_A.T @ X_A
            G_A_inv = G_A.inverse()
            I_A = torch.ones_like(A).int()
            import pdb; pdb.set_trace()
            A_A = G_A_inv.sum(dim = 0).sum(dim = 0) ** (-1/2)
            w_A = A_A * G_A_inv.sum(dim = 1).view(-1, 1)
            u_A = X_A @ w_A
            import pdb; pdb.set_trace()
            a = X.T @ u_A
            gammas = ...
            j = gammas.argmin()
            gamma = gammas[j]
            mu_hat += gamma * u_A
            A[j] = True

