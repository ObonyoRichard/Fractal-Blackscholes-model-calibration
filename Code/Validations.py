'''
This module defines the various validation methods which are usually run at the end of every epoch        
'''

import torch

def validate(model, val_loader, loss_fn):
  ''' Method which accepts the model as well as the validation data loader and logs the validation loss '''
  loss_val = 0.0
  with torch.no_grad():
    for Xs, Ys in val_loader:
      Xs = Xs.to(device=device)
      Ys = Ys.float().to(device=device)        
      outputs = model(Xs)
      loss_val += loss_fn(outputs, Ys.view(-1,1)).item()
  print(f"Validation Loss {round(loss_val/len(val_loader),5)}")

def iValidate(model, val_loader, loss_fn):
  ''' Method which accepts the model as well as train and validation data loaders and logs the accuracy '''
  loss_val = 0.0
  with torch.no_grad():
    for Xs, Ys in val_loader:
      Xs = Xs.to(device=device)
      Ys = Ys.to(device=device)        
      outputs = model(Xs)
      loss_val += loss_fn(outputs, Ys).item()
  print(f"Validation Loss {round(loss_val/len(val_loader),5)}")