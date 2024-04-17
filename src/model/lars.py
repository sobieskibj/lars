import torch
from torch import nn
import torch.nn.functional as F

import logging
log = logging.getLogger(__name__)

class LARS(nn.Module):

    def __init__(self, p: int):
        super().__init__()
        self.p = p
        self.betas = torch.zeros(p, p)

    def fit(self, dataset):
        X, y = dataset.get_data()

        mu_hat = torch.zeros_like(y)
        s = torch.zeros_like(X[0, None].T)
        A = torch.zeros_like(s).bool()

        # In the beginning, we choose the predictor
        # according to highest absolute correlation
        c_hat = X.T @ y
        A[c_hat.abs().argmax()] = True

        for k in range(self.p - 1):
            log.info(f'Iteration: {k}')

            # Eq. 2.8
            c_hat = X.T @ (y - mu_hat)

            # Eq. 2.9
            C_hat = c_hat.abs().max()

            s[A] = c_hat[A].sign()

            # Eq. 2.4
            X_A = (X * s.T)[:, A.flatten()]

            # Eq. 2.5
            G_A = X_A.T @ X_A
            G_A_inv = G_A.inverse()
            A_A = G_A_inv.sum(dim = 0).sum(dim = 0) ** (-1/2)

            if not k == self.p - 2:

                # Eq. 2.6
                w_A = A_A * G_A_inv.sum(dim = 1).view(-1, 1)
                u_A = X_A @ w_A

                # Eq. 2.11
                a = X.T @ u_A

                # Eq. 2.13

                # Compute pairs of gammas for each predictor
                gammas_l = (C_hat - c_hat) / (A_A - a)
                gammas_r = (C_hat + c_hat) / (A_A + a)

                # Set to infinity any element that corresponds
                # to active set or is negative
                gammas_l[A.logical_or(gammas_l <= 0)] = torch.inf
                gammas_r[A.logical_or(gammas_r <= 0)] = torch.inf

                # Get gamma for each predictor by taking minimum
                # element from each pair
                gammas, _ = torch.hstack([gammas_l, gammas_r]).min(dim = 1)

                # Find the minimizer and its gamma
                j = gammas.argmin()
                gamma = gammas[j]

            else:
                # At last iteration, the best predictor is known
                # and the formula for step size is simplified
                j = A.int().argmin()
                gamma = C_hat / A_A
            
            # Eq. 2.14
            mu_hat += gamma * u_A

            # Add minimizer to active set
            log.info(f'MSE: {F.mse_loss(y, mu_hat)}')

            A[j] = True

