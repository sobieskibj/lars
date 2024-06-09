import time
import wandb
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base import Model

import logging
log = logging.getLogger(__name__)

class LARS(Model):

    def __init__(self, speedup):
        super().__init__()
        self.speedup = speedup

    def setup(self, X, device):
        p = X.shape[1]
        self.p = p
        self.betas = torch.zeros(p + 1, p, device = device)
        self.alphas = torch.zeros(p + 1, device = device)

    def update_betas(self, k, s, A, gamma, w_A, X, y):
        if k != self.p - 1:
            # Eq. 3.3
            s_w_A = s.clone()
            s_w_A[A.flatten(), :] *= w_A
            betas_delta = gamma * s_w_A.flatten()
            self.betas[k + 1] = self.betas[k] + betas_delta
        else:
            # In the end, the coefficients are equal to OLS solution
            self.betas[k + 1] = ((X.T @ X).inverse() @ X.T @ y).flatten()

    def update_alphas(self, k, X, y):
        self.alphas[k] = torch.hstack([X, y]).T.cov()[-1, :-1].abs().max()

    def train(self, dataset):
        log.info('Training')

        device = torch.device('cuda') if \
            torch.cuda.is_available() else torch.device('cpu')
        
        # Extract training split
        X, y, _, _ = dataset.get_train_val_split()
        X, y = X.to(device), y.to(device)

        # Prepare model using X
        self.setup(X, device)

        mu_hat = torch.zeros_like(y)
        s = torch.zeros_like(X[0, None].T)
        A = torch.zeros_like(s).bool()

        # Start monitoring execution time
        start_time = time.time()

        # In the beginning, we choose the predictor
        # according to highest absolute correlation
        c_hat = X.T @ y
        A[c_hat.abs().argmax()] = True

        for k in range(self.p - 1):
            if not self.speedup: log.info(f'Iteration: {k}')

            # Eq. 2.8
            c_hat = X.T @ (y - mu_hat)
            if not self.speedup: self.update_alphas(k, X, (y - mu_hat))

            # Eq. 2.9
            C_hat = c_hat.abs().max()

            s[A] = c_hat[A].sign()

            # Eq. 2.4
            X_A = (X * s.T)[:, A.flatten()]

            # Eq. 2.5
            G_A = X_A.T @ X_A
            G_A_inv = G_A.inverse()
            A_A = G_A_inv.sum() ** (-1/2)

            # Eq. 2.6
            w_A = A_A * G_A_inv.sum(dim = 1).view(-1, 1)
            u_A = X_A @ w_A

            if not k == self.p - 2:
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

            # Update coefficients
            if not self.speedup: self.update_betas(k, s, A, gamma, w_A, X, y)

            # Add minimizer to active set
            A[j] = True

            if not self.speedup:
                # Compute mse
                loss = F.mse_loss(y, mu_hat)
                log.info(f'MSE: {loss.item()}')

        if not self.speedup: 
            log.info(f'Iteration: {k + 1}')

            # OLS solution
            self.update_betas(k + 1, s, A, gamma, w_A, X, y)

            # Get residuals at last iter equal to OLS
            mu_hat = (X @ self.betas[-1])[:, None]
            self.update_alphas(k + 1, X, (y - mu_hat))

            # Compute mse
            log.info(f'MSE: {F.mse_loss(y, mu_hat)}')

        # End monitoring execution time
        end_time = time.time()

        wandb.log({'exec_time': end_time - start_time})

    def validate(self, dataset):
        log.info('Validation')

        device = torch.device('cuda') if \
            torch.cuda.is_available() else torch.device('cpu')

        # Extract validation split
        _, _, X, y = dataset.get_train_val_split()
        X, y = X.to(device), y.to(device)

        for iter_idx, betas in enumerate(self.betas):
            mu_hat = (X @ betas)[:, None]
            loss = F.mse_loss(y, mu_hat).item()
            log.info(f'Iteration: {iter_idx}')
            log.info(f'MSE: {loss}')
            wandb.log({'loss': loss})

        if not self.speedup:
            self.log_beta_plot()

    def log_beta_plot(self):
        plt.figure()
        plt.plot(
            self.alphas.flip(0), self.betas.flip(0).numpy(force = True))
        plt.xlabel('Alpha')
        plt.ylabel('Betas')
        plt.legend([idx for idx in range(self.p + 1)])
        plt.grid()
        wandb.log({'betas': plt})