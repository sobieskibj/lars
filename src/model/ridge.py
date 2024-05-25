import wandb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge as sk_Ridge
import matplotlib.pyplot as plt

from .base import Model

import logging
log = logging.getLogger(__name__)

class Ridge(Model):

    def __init__(self, min_lambda: float, max_lambda: float, n_lambdas: int):
        super().__init__()

        self.lambdas = np.geomspace(min_lambda, max_lambda, num = n_lambdas)

    def train(self, dataset):
        log.info('Training')

        # Extract training split
        X, y, _, _ = dataset.get_train_val_split()

        # Make container for betas
        self.betas = np.zeros((len(self.lambdas) + 1, X.shape[1]))

        for iter_idx, lambda_ in enumerate(self.lambdas):
            log.info(f'Iteration: {iter_idx}')

            # Instantiate and train model with specific lambda
            model = sk_Ridge(alpha = lambda_, fit_intercept = False)    
            model.fit(X, y)

            # Save betas
            self.betas[iter_idx + 1] = model.coef_

            # Get predictions and mse
            y_hat = model.predict(X)
            log.info(f'MSE: {mean_squared_error(y, y_hat)}')

    def get_alpha(self, X, y):
        return np.absolute(np.cov(X.T, y.flatten())[-1, :-1]).max()

    def validate(self, dataset):
        log.info('Validation')

        # Extract validation split
        _, _, X, y = dataset.get_train_val_split()
        p = self.betas.shape[1]
        alphas = np.zeros(self.betas.shape[0])

        for iter_idx, betas in enumerate(self.betas):
            log.info(f'Iteration: {iter_idx}')

            # Get predictions and mse with coefs for each lambda
            y_hat = X @ betas[:, None]
            alphas[iter_idx] = self.get_alpha(X, (y - y_hat))
            loss = mean_squared_error(y, y_hat)
            log.info(f'MSE: {loss}')
            wandb.log({'loss': loss})

        plt.figure()
        plt.plot(alphas[::-1], self.betas[::-1])
        plt.xlabel('Alpha')
        plt.ylabel('Betas')
        plt.legend([idx for idx in range(p + 1)])
        plt.grid()
        wandb.log({'betas': plt})