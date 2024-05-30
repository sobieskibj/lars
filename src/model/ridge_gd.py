import time
import wandb
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from .base import Model

import logging
log = logging.getLogger(__name__)

class RidgeGD(Model):

    def __init__(
            self, 
            min_lambda: float, 
            max_lambda: float, 
            n_lambdas: int,
            optimizer: torch.optim.Optimizer,
            n_epochs: int,
            early_stop: int):
    
        super().__init__()
    
        self.lambdas = torch.from_numpy(
            np.geomspace(min_lambda, max_lambda, num = n_lambdas))
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.early_stop = early_stop

    def setup(self, X):
        p = X.shape[1]
        self.p = p
        self.betas = torch.zeros(p + 1, p).float()

    def fit(self, X, y, lambda_):
        log.info(f'Fitting model for lambda={lambda_}')

        # Setup parameters and optimizer
        betas = nn.Parameter(data = torch.zeros(self.p, 1))
        optim = self.optimizer(params = [betas])

        # Make constants for tracking progress
        prev_loss = torch.tensor(float('inf'))
        stop_counter = 0

        # Iterate self.n_epochs times
        pbar = tqdm(range(self.n_epochs))

        for epoch_idx in pbar:
            log.info(f'Epoch: {epoch_idx}')
            optim.zero_grad()

            # Compute loss and backward pass
            y_hat = X @ betas
            loss = F.mse_loss(y_hat, y) + lambda_ * betas.norm(p = 2)
            loss.backward()

            if prev_loss < loss:
                stop_counter += 1

                if stop_counter == self.early_stop:
                    break

            else:
                stop_counter = 0

            prev_loss = loss.item()

            # Make gradient step
            optim.step()

            pbar.set_description(f'loss: {loss.item()}')

        return betas

    def train(self, dataset):
        log.info('Training')

        # Extract training split
        X, y, _, _ = dataset.get_train_val_split()

        # Prepare model using X
        self.setup(X)
        start_time = time.time()

        # Iterate over lambdas and find coefs for each
        for iter_idx, lambda_ in enumerate(self.lambdas):
            log.info(f'Iteration: {iter_idx}')

            # Fit model
            betas = self.fit(X, y, lambda_)

            # Save coefs
            self.betas[iter_idx + 1] = betas.flatten()

            # Get predictions and mse
            y_hat = X @ betas.view(-1, 1)
            log.info(f'MSE: {F.mse_loss(y, y_hat)}')

        end_time = time.time()
        wandb.log({'exec_time': end_time - start_time})

    def get_alpha(self, X, y):
        return torch.cat([X.T, y.T]).cov()[-1, :-1].abs().max().item()

    def validate(self, dataset):
        log.info('Validation')

        # Extract validation split
        _, _, X, y = dataset.get_train_val_split()
        alphas = torch.zeros(self.betas.shape[0])

        for iter_idx, betas in enumerate(self.betas):
            mu_hat = (X.float() @ betas)[:, None]
            alphas[iter_idx] = self.get_alpha(X, (y - mu_hat))
            loss = F.mse_loss(y, mu_hat).item()
            log.info(f'Iteration: {iter_idx}')
            log.info(f'MSE: {loss}')
            wandb.log({'loss': loss})

        plt.figure()
        plt.plot(
            alphas.flip(0), self.betas.flip(0).numpy(force = True))
        plt.xlabel('Alpha')
        plt.ylabel('Betas')
        plt.legend([idx for idx in range(self.p + 1)])
        plt.grid()
        wandb.log({'betas': plt})