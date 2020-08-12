'''
This module handles all the torch DataSet classes needed for training various neural networks        
'''

import torch
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset

class fBMDataset(Dataset):
    
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index, :]
        X = torch.from_numpy(item[:-1].values).float()
        Y = item[-1]
        return X, Y

class fBMDatasetWithK(Dataset):
    
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data.insert(0,"K",np.random.normal(0,1,len(self.data)))
        self.data["S"] = self.data["S"]*self.data["K"]
        self.data["C"] = (self.data["C"] + 0.5)*self.data["K"] - 0.5
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index, :]
        X = torch.from_numpy(item[:-1].values).float()
        Y = item[-1]
        return X, Y

class fBMDatasetInverseChain(Dataset):
    
    def __init__(self, data_path, cann_path, inverse_path ):
        original = pd.read_csv(data_path)[["S","r","K","τ"]]
        cann = pd.read_csv(cann_path)
        self.inverse = pd.read_csv(inverse_path)
        self.data = pd.concat([cann, original], axis=1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index, :]
        X = torch.from_numpy(item.values).float()
        Y = torch.tensor(self.inverse.iloc[index]).float()
        return X, Y

class fBMDatasetInference(Dataset):
    
    def __init__(self, testData ):
        self.data = testData[['C','S','r','K','τ']]
        self.data = (self.data - tMeans)/tStds
        self.data['C'] = self.data['C'] - minC - 0.5

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index, :]
        X = torch.from_numpy(item.values).float()
        return X, X