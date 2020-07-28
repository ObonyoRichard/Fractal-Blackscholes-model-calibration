'''
This gateway module defines the inverse map with MSE loss function for predicting model parameters
and its training loop        
'''

import torch
from Validations import iValidate
from DataLoading import fBMDatasetInverse
from ModelVectors import trainFile, valFile
from Diagnostics import iPlotPredictions, iGetTrainAndValOutputs

#------------------ Parameters -----------------------

ModelLoadPath = "----------"
trainPredictionPath = "---------"
valPredictionPath = "----------"

#----------------- Helper functions ------------------

def i_training_loop_mse(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
  ''' Method to execute the training '''
  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    for Xs, Ys in train_loader:
      Xs = Xs.to(device=device)
      Ys = Ys.to(device=device)
      outputs = model(Xs)
      loss = loss_fn(outputs, Ys)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_train += loss.item()
    print(f'{datetime.datetime.now()} Epoch {epoch} \nTraining Loss {round(loss_train/len(train_loader),5)}')
    iValidate(model, val_loader, loss_fn)

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")

#----------- Model and Dataset definition -----------------

trainDataSet = fBMDatasetInverse(trainFile,trainPredictionPath)
valDataSet = fBMDatasetInverse(valFile,valPredictionPath)
model = Pricer(4,2).to(device=device)
logModelDetails(model)
loss_fn = nn.MSELoss()

model.load_state_dict(torch.load("<ModelLoadPath>", map_location=device))

#------- Train or Load the saved weights ------------------

trainLoader = torch.utils.data.DataLoader(trainDataSet, 64, shuffle= True)
valLoader = torch.utils.data.DataLoader(valDataSet, 64, shuffle= True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#-------- Plot the predictions -----------------------------

yActualTrain, yActualVal, yPredTrain, yPredVal = iGetTrainAndValOutputs( model, trainLoader, valLoader)
iPlotPredictions(yActualTrain, yActualVal, yPredTrain, yPredVal)