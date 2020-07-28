'''
This module exposes a method for getting the inferences as well as a method for plotting 
in-sample and out-of-sample predictions       
'''

import numpy as np
import torch
import matplotlib as plt

def getTrainAndValOutputs( model, train_loader, val_loader ):
  yActualTrain, yActualVal = np.zeros(24000), np.zeros(6000)
  yPredTrain, yPredVal = np.zeros(24000), np.zeros(6000)
  with torch.no_grad():
    model.eval()
    i = 0
    for Xs, Ys in train_loader:
      Xs = Xs.to(device=device)
      yPredTrain[i:i+len(Ys)] = model(Xs).numpy().flatten()
      yActualTrain[i:i+len(Ys)] = Ys.numpy().flatten()
      i += len(Ys)
    i = 0
    for Xs, Ys in val_loader:
      Xs = Xs.to(device=device)
      yPredVal[i:i+len(Ys)] = model(Xs).numpy().flatten()
      yActualVal[i:i+len(Ys)] = Ys.numpy().flatten()
      i += len(Ys)
  return yActualTrain, yActualVal, yPredTrain, yPredVal

def plotPredictions(yActualTrain, yActualVal, yPredTrain, yPredVal):
  ''' Plot both train and validation predictions '''
  plt.figure(figsize=(6, 3))
  plt.subplot(131)
  plt.scatter(yActualTrain, yPredTrain, s=1)
  plt.subplot(132)
  plt.scatter(yActualVal,yPredVal, s=1)
  plt.show()

def iGetTrainAndValOutputs( model, train_loader, val_loader ):
  yActualTrain, yActualVal = np.zeros((24000,2)), np.zeros((6000,2))
  yPredTrain, yPredVal = np.zeros((24000,2)), np.zeros((6000,2))
  with torch.no_grad():
    model.eval()
    i = 0
    for Xs, Ys in train_loader:
      Xs = Xs.to(device=device)
      yPredTrain[i:i+len(Ys)] = model(Xs).numpy()
      yActualTrain[i:i+len(Ys)] = Ys.numpy()
      i += len(Ys)
    i = 0
    for Xs, Ys in val_loader:
      Xs = Xs.to(device=device)
      yPredVal[i:i+len(Ys)] = model(Xs).numpy()
      yActualVal[i:i+len(Ys)] = Ys.numpy()
      i += len(Ys)
  return yActualTrain, yActualVal, yPredTrain, yPredVal


def iPlotPredictions(yActualTrain, yActualVal, yPredTrain, yPredVal):
  ''' Plot both train and validation predictions '''
  #  plt.figure(figsize=(18, 12))
  plt.figure()
  plt.subplot(2, 2, 1)
  plt.scatter(yActualTrain[:,0], yPredTrain[:,0], s=1)
  plt.xlabel("Actual Values")
  plt.ylabel("Predicted Values")
  plt.title("In sample")
  plt.subplot(2, 2, 2)
  plt.scatter(yActualVal[:,0],yPredVal[:,0], s=1)
  plt.xlabel("Actual Values")
  plt.ylabel("Predicted Values")
  plt.title("Out of sample")
  plt.subplot(2, 2, 3)
  plt.scatter(yActualTrain[:,1], yPredTrain[:,1], s=1)
  plt.xlabel("Actual Values")
  plt.ylabel("Predicted Values")
  plt.title("In sample")
  plt.subplot(2, 2, 4)
  plt.scatter(yActualVal[:,1],yPredVal[:,1], s=1)
  plt.xlabel("Actual Values")
  plt.ylabel("Predicted Values")
  plt.title("Out of sample")
  plt.show()