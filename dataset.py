import pandas as pd
import numpy as np
from scipy.stats import zscore
import os

def _splitext(x):
    return os.path.splitext(x)[0]

def _centered(x):
    return x - np.mean(x)

class Dataset:
    def __init__(self):
        self._files = os.listdir('dataset')
        if '.DS_Store' in self._files: self._files.remove('.DS_Store')

    def rx(self, k):
        data = {}
        for i in range(len(self._files)):
            x = pd.read_csv(f'dataset/{self._files[i]}', encoding='cp949')
            y = x.groupby('Unnamed: 0').get_group(f'Slave{k}')
            y = y.drop('Unnamed: 0', axis=1)
            y = y.T.apply(_centered).T
            y.index = [f'{i}' for i in range(len(y))]
            data[_splitext(self._files[i])] = y

        return data

    def pulseint(self, k, method='noncoherent'):
        data = self.rx(k)

class Param:
    def __init__(self, nsamples=512, fs=42e9):
        self.n = 512
        self.fs = 42e9
        self.ts = 1 / fs
        self.t = np.linspace(0, self.n, self.n, endpoint=False) * self.ts