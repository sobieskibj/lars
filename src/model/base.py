from torch import nn
from abc import ABC, abstractmethod

class Model(ABC, nn.Module):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass