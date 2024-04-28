import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

from .base import Model

import logging
log = logging.getLogger(__name__)

class LASSO(Model):

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
            model = Lasso(alpha = lambda_, fit_intercept = False)    
            model.fit(X, y)

            # Save betas
            self.betas[iter_idx + 1] = model.coef_

            # Get predictions and mse
            y_hat = model.predict(X)
            log.info(f'MSE: {mean_squared_error(y, y_hat)}')

    def validate(self, dataset):
        log.info('Validation')

        # Extract validation split
        _, _, X, y = dataset.get_train_val_split()

        for iter_idx, betas in enumerate(self.betas):
            log.info(f'Iteration: {iter_idx}')

            # Get predictions and mse with coefs for each lambda
            y_hat = X @ betas[:, None]
            log.info(f'MSE: {mean_squared_error(y, y_hat)}')