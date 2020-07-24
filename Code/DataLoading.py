import numpy as np 
import pandas as pd

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