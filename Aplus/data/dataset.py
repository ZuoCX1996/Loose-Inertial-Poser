import pandas as pd
import numpy as np
import time
from torch.utils.data import Dataset
import torch

def random_index(data_len:int, sample_rate=1.0, seed:int=None) -> list:
    """
    Generate random index according len of data.
    Args:
        data_len: Length of data.
        sample_rate: Proportion of data be selected. Default = 1
        seed: Random seed, if None, using current time.

    Returns:
        index: list
        Random selected index list with len of [data_len * sample_rate]
    """
    index = [i for i in range(data_len)]
    df_index = pd.DataFrame({'index': index})
    if seed is not None:
        rand_select = np.array(df_index.sample(frac=sample_rate, random_state=seed)['index']).tolist()
    else:
        rand_select = np.array(df_index.sample(frac=sample_rate, random_state=int(time.time()))['index']).tolist()

    return rand_select

def data_shuffle(*args, seed=None):
    """
    Shuffle data at dim 0.
    Args:
        *args: Data with same length. Can be [torch.Tensor] or [ndarray].
        seed: Random seed, if None, using current time.

    Returns:
        Shuffled data. When multiple data was given as input, it will be tuple.

    """
    data = list(args)
    rand_index = random_index(data_len=len(data[0]), seed=seed)
    for i, d in enumerate(data):
        data[i] = data[i][rand_index]
    data = tuple(data)
    return data


class BaseDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        self.data_len = len(x)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = (index % self.data_len)
        return self.x[i], self.y[i]
    @staticmethod
    def load_data(path:str) -> dict :
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files

        Returns:
            Dict of datas.
        """
        pass



