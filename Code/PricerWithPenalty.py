'''
This gateway module defines the pricer with MSE loss function as well as the soft constraints
and its training loop        
'''

import torch
from Validations import validate
from DataLoading import fBMDatasetWithK
from ModelVectors import trainFile, valFile
from Diagnostics import plotPredictions, getTrainAndValOutputs

#------------------ Parameters -----------------------

ModelLoadPath = "----------"

#----------------- Helper functions ------------------

def pf(x, λ=1, m=3):
  ''' Penalty function for applying soft constraints of no arbitrage conditions '''
  fn = λ*(x**m)
  return fn*(fn>0)

def training_loop_penalty(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
  ''' Method to execute the training '''
  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    for Xs, Ys in train_loader:
      Xs = Variable(Xs.to(device=device), requires_grad=True)
      Ys = Variable(Ys.float().to(device=device), requires_grad=True)
      outputs = model(Xs)
      dCdx = torch.autograd.grad(outputs, Xs, grad_outputs=outputs.data.new(outputs.shape).fill_(1), create_graph=True, retain_graph=True)[0]
      d2Cdx2 = torch.autograd.grad(dCdx, Xs, grad_outputs=dCdx.data.new(dCdx.shape).fill_(1), create_graph=True, retain_graph=True)[0]
      K, T = Xs[:,0], Xs[:,3]
      dCdk, dCdt = dCdx[:,0], dCdx[:,3]
      d2Cdk2 = d2Cdx2[:,0]
      penalty = torch.sum(pf(-(K**2)*(d2Cdk2))) + torch.sum(pf(-T*dCdt)) + torch.sum(pf(K*dCdk))
      mse = loss_fn(outputs, Ys.view(-1,1))
      loss = mse + penalty
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_train += loss.item()
    print(f'{datetime.datetime.now()} Epoch {epoch} \nTraining Loss {round(loss_train/len(train_loader),5)}')
    validate(model, val_loader, loss_fn)

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")

#----------- Model and Dataset definition -----------------

trainDataSet = fBMDatasetWithK(trainFile)
valDataSet = fBMDatasetWithK(valFile)
model = Pricer(6,1).to(device=device)
logModelDetails(model)
loss_fn = nn.MSELoss()

model.load_state_dict(torch.load("<ModelLoadPath>", map_location=device))

#------- Train or Load the saved weights ------------------

trainLoader = torch.utils.data.DataLoader(trainDataSet, 64, shuffle= True)
valLoader = torch.utils.data.DataLoader(valDataSet, 64, shuffle= True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#-------- Plot the predictions -----------------------------

yActualTrain, yActualVal, yPredTrain, yPredVal = getTrainAndValOutputs( model, trainLoader, valLoader)
plotPredictions(yActualTrain, yActualVal, yPredTrain, yPredVal)