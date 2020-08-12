'''
This module is to compute and store random vectors sampling the fractional Black Scholes 
Model for call options with the following characteristics.
Stock price by Strike price (S) 0.6   - 1.4
Maturity                    (T) 1 day - 3 years
Risk free rate              (r) 1%    - 7%
Volatility                  (σ) 2%    - 90%
Hurst Parameter             (H) 0     - 1            
'''

import numpy as np
import pandas as pd
from math import exp
from math import log as ln
from scipy.stats import norm
from random import random

#-------------------- Parameters ---------------------------

trainFile = "----"
valFile   = "----"
msFile    = "----"
totalSize = 30000
split     = int(0.8*totalSize)

#----------------- Helper functions ------------------------

randS = lambda : 0.6 + random()*0.8
randτ = lambda : 1/365 + random()*1094/365
randR = lambda : 0.01 + random()*0.06
randσ = lambda : 0.02 + random()*0.88
randH = lambda : random()

Φ = norm.cdf

d1 = lambda S,K,r,τ,σ,H : (ln(S/K) + r*τ + 0.5*(σ**2)*(τ**(2*H)))/(σ*(τ**H))

d2 = lambda S,K,r,τ,σ,H : d1(S,K,r,τ,σ,H) - σ*(τ**H)

def Cf(S,K,r,τ,σ,H):    
  ''' fractional Black-Scholes value of a European call option '''
  return S*Φ(d1(S,K,r,τ,σ,H)) - (K*exp(-r*τ))*Φ(d2(S,K,r,τ,σ,H))

def createData( dataSize ):
  ''' Create <dataSize> number of fBSM entries in the <dataFile> '''
  i, data = 0, np.zeros((30000,6))
  while i < dataSize:
    S,r,τ,σ,H = randS(),randR(),randτ(),randσ(),randH()
    C = Cf(S,1,r,τ,σ,H)
    if C < 0.01: continue
    data[i,:] = [S,r,τ,σ,H,C]
    i += 1
  return data

def createDataFile( dataFile, data ):
  ''' Create <dataSize> number of fBSM entries in the <dataFile> '''
  pd.DataFrame(data).to_csv(dataFile, header = ["S","r","τ","σ","H","C"], index = None)
  print(f"File <{dataFile}> created with {len(data)} entries")

def createTestData( dataSize, H, σ ):
  ''' Create <dataSize> number of fBSM entries in the <dataFile> '''
  i, data = 0, np.zeros((dataSize,7))
  while i < dataSize:
    SbyK,K,r,τ = randSbyK(),randK(),randR(),randτ()
    S = SbyK*K
    C = Cf(S,K,r,τ,σ,H)
    if C < 0.001: continue
    data[i,:] = [S,K,r,τ,σ,H,C]
    i += 1
  return pd.DataFrame(data, columns=columns)

#----------------- The main activity --------------------------

data = createData( totalSize )
means = np.array(list(map(lambda i: data[:,i].mean(), range(0,6))))
stds = np.array(list(map(lambda i: data[:,i].std(), range(0,6))))
normData = (data - means)/stds
minCost = min(normData[:,-1]) 
print(minCost) #-1.472140313637543
normData[:,-1] = normData[:,-1] - minCost - 0.5

createDataFile( trainFile, normData[:split])
createDataFile( valFile, normData[split:])
createDataFile( msFile, np.array([means,stds]) )