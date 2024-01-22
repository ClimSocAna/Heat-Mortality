import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    
    '''
        A dummy dataset for dataloader, n controls the size of 1 epoch.
    '''

    def __init__(self, n=100):
        self.n = n
        
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor(0)