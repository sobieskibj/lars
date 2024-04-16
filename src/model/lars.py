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

        import pdb; pdb.set_trace()


