from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseDataset(ABC, Dataset):

    @abstractmethod
    def get_train_val_split():
        pass

    