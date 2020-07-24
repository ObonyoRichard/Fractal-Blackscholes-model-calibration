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

yActualTrain, yActualVal, yPredTrain, yPredVal = getTrainAndValOutputs( model, trainLoader, valLoader)

def plotPredictions(yActualTrain, yActualVal, yPredTrain, yPredVal):
  ''' Plot both train and validation predictions '''
  plt.figure(figsize=(6, 3))
  plt.subplot(131)
  plt.scatter(yActualTrain, yPredTrain, s=1)
  plt.subplot(132)
  plt.scatter(yActualVal,yPredVal, s=1)
  plt.show()