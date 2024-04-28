import torch
import numpy as np
from sklearn.datasets import load_diabetes

from .base import BaseDataset

class Diabetes(BaseDataset):

    def __init__(self, split: float):
        super().__init__()
        self.split = split
        self.setup_data()

    def setup_data(self):
        # load diabetes dataset
        data = load_diabetes(as_frame = True)

        # center and normalize X and y
        X = data['data'].values
        X -= X.mean(axis = 0) 
        X /= np.linalg.norm(X, axis = 0)
        self.X = torch.tensor(X).float()

        y = data['target'].values
        y -= y.mean(axis = 0) 
        y /= np.linalg.norm(y, axis = 0)
        self.y = torch.tensor(y).float().view(-1, 1)

        # get length
        self.n = X.shape[0]

    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        output = {
            'x': self.X[index],
            'y': self.y[index]}
        return output
    
    def get_train_val_split(self):
        split_idx = round(self.n * self.split)
        X_train, y_train = self.X[:split_idx], self.y[:split_idx]
        X_val, y_val = self.X[split_idx:], self.y[split_idx:]
        return X_train, y_train, X_val, y_val
    
    def get_data(self):
        return self.X, self.y