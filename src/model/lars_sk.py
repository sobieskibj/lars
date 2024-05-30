import time
import wandb
import torch
from sklearn.linear_model import Lars
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base import Model

import logging
log = logging.getLogger(__name__)

class LARSSK(Model):

    def __init__(self):
        super().__init__()

    def train(self, dataset):
        log.info('Training')

        # Extract training split
        X, y, _, _ = dataset.get_train_val_split()

        # Create model and fit
        model = Lars(fit_intercept = False, precompute = False)

        start_time = time.time()

        model.fit(X, y)

        end_time = time.time()
        
        wandb.log({'exec_time': end_time - start_time})
        
        # Save model parameters
        self.model = model
        self.alphas = model.alphas_
        self.betas = model.coef_path_.T
        self.order = model.active_

    def validate(self, dataset):
        log.info('Validation')

        # Extract validation split
        _, _, X, y = dataset.get_train_val_split()

        for iter_idx, betas in enumerate(self.betas):
            mu_hat = (X @ betas)[:, None]
            loss = F.mse_loss(y, mu_hat).item()
            log.info(f'Iteration: {iter_idx}')
            log.info(f'MSE: {loss}')
            wandb.log({'loss': loss})

        self.log_alpha_beta_plot()


    def log_alpha_beta_plot(self):
        plt.figure()
        plt.plot(self.alphas, self.betas)
        plt.xlabel('Alpha')
        plt.ylabel('Betas')
        plt.legend([idx for idx, _ in enumerate(self.order)])
        plt.grid()
        wandb.log({'betas': plt})