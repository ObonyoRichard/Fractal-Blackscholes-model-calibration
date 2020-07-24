import torch.nn as nn
import torch.nn.functional as F

def logModelDetails(model):
  ''' log a Pytorch model specs '''
  print(model)
  params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Total trainable paramters: {params}")

class Pricer (nn.Module):
  ''' The pricer neural network functional '''

  def __init__(self,inLength,outLength):
    super().__init__()
    self.fc1 = nn.Linear(inLength,128) 
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 128)
    self.final = nn.Linear(128,outLength)

  def forward(self, x):
    out = self.fc1(x)
    out = F.dropout(out, p = 0.2)
    out = F.gelu(self.fc2(out))
    out = F.dropout(out, p = 0.2)
    out = F.gelu(self.fc3(out))
    out = F.dropout(out, p = 0.2)
    out = F.softplus(self.final(out)) - 0.5
    return out