'''
This gateway module defines the Vanilla pricer with MSE loss function and its training loop        
'''

import torch
from Validations import validate
from DataLoading import fBMDataset
from ModelVectors import trainFile, valFile
from Diagnostics import plotPredictions, getTrainAndValOutputs

#------------------ Parameters -----------------------

ModelLoadPath = "----------"

#----------------- Helper functions ------------------

def training_loop_mse(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
  ''' Method to execute the training '''
  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    for Xs, Ys in train_loader:
      print(Xs.requires_grad)
      Xs = Xs.to(device=device)
      Ys = Ys.float().to(device=device)
      outputs = model(Xs)
      loss = loss_fn(outputs, Ys.view(-1,1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_train += loss.item()
    print(f'{datetime.datetime.now()} Epoch {epoch} \nTraining Loss {round(loss_train/len(train_loader),5)}')
    validate(model, val_loader, loss_fn)

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")

#----------- Model and Dataset definition -----------------

trainDataSet = fBMDataset(trainFile)
valDataSet = fBMDataset(valFile)
model = Pricer(5,1).to(device=device)
logModelDetails(model)

#------- Train or Load the saved weights ------------------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainLoader = torch.utils.data.DataLoader(trainDataSet, 64, shuffle= True)
valLoader = torch.utils.data.DataLoader(valDataSet, 64, shuffle= True)
loss_fn = nn.MSELoss()

model.load_state_dict(torch.load("<ModelLoadPath>", map_location=device))

#-------- Plot the predictions -----------------------------

yActualTrain, yActualVal, yPredTrain, yPredVal = getTrainAndValOutputs( model, trainLoader, valLoader)
plotPredictions(yActualTrain, yActualVal, yPredTrain, yPredVal)