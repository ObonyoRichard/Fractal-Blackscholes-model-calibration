'''
This module executes the preliminary inferrence procedure using log transform proposal      
'''

import torch
import numpy as np
from ModelVectors import createTestData
from Dataloading import fBMDatasetInference
from sklearn.linear_model import LinearRegression

tMeans = [0.15630499,0.54902774,0.04021916,0.53220327,1.57032268]

tStds = [0.13603969,0.30926215,0.01731955,0.27354582,0.84257395]

minC = -1.0754482995211179
minOutput = -1.51383233733901
μOutput = 0.6039890110123125
σOutput = 0.3986395125288202


def infer( model, loader, length ):
  lnT, predictions,cost = np.zeros(length), np.zeros(length), np.zeros(length)
  with torch.no_grad():
    model.eval()
    i = 0
    for Xs, _ in loader:
      Xs = Xs.to(device=device)
      predictions[i:i+len(Xs)] = model(Xs).numpy().flatten()
      lnT[i:i+len(Xs)] = Xs[:,-1]
      cost[i:i+len(Xs)] = Xs[:,0]
      i += len(Xs)
  return lnT, predictions

#---------------- Generating test data ------------------------

testLength, H, σ = 1000, 0.6, 0.1
testData = createTestData(testLength, H, σ )

marketData = fBMDatasetInference(testData)
mdLoader = torch.utils.data.DataLoader(marketData, 1, shuffle= False)

#---------------- Log Transform + Linear Regression -----------

timePeriod, predictions = infer(iModel, mdLoader, testLength)
Y = (predictions+minOutput+0.5)*σOutput + μOutput
X = timePeriod*tStds[-1] + tMeans[-1]
regression = LinearRegression().fit(np.log(X).reshape(-1,1),np.log(Y))
print(np.exp(regression.intercept_))
print(regression.coef_[0])

timePeriod, predictions = infer(icModel, mdLoader, testLength)
cY = (predictions+minOutput+0.5)*σOutput + μOutput
cX = timePeriod*tStds[-1] + tMeans[-1]
regression = LinearRegression().fit(np.log(cX).reshape(-1,1),np.log(cY))
print(np.exp(regression.intercept_))
print(regression.coef_[0])