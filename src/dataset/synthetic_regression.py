import torch
from torch.utils.data import Dataset

class SyntheticRegressionDataset(Dataset):

    def __init__(self, n: int, p: int, split: float):
        super().__init__()
        self.n, self.p = n, p
        self.split = split
        self.setup_data()

    def setup_data(self):
        X = torch.randn(self.n, self.p)
        X -= X.mean(dim = 0) 
        X /= X.norm(dim = 0)
        self.X = X
        self.beta = torch.randn(self.p, 1)
        self.y = self.X @ self.beta

    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        output = {
            'x': self.X[index],
            'y': self.y[index]}
        return output
    
    def get_true_beta(self):
        return self.beta
    
    def get_train_val_split(self):
        split_idx = round(self.n * self.split)
        X_train, y_train = self.X[:split_idx], self.y[:split_idx]
        X_val, y_val = self.X[split_idx:], self.y[split_idx:]
        return X_train, y_train, X_val, y_val
    
    def get_data(self):
        return self.X, self.y