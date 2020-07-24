import torch
from Code.DataLoading import fBMDataset
from Code.ModelVectors import trainFile, valFile

#------------------ Parameters -----------------------

ModelLoadPath = "----------"

#----------------- Helper functions ------------------

def validate(model, val_loader):
  ''' Method which accepts the model as well as the validation data loader and logs the validation loss '''
  loss_val = 0.0
  with torch.no_grad():
    for Xs, Ys in val_loader:
      Xs = Xs.to(device=device)
      Ys = Ys.float().to(device=device)        
      outputs = model(Xs)
      loss_val += loss_fn(outputs, Ys.view(-1,1)).item()
  print(f"Validation Loss {round(loss_val/len(val_loader),5)}")

def training_loop_mse(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
  ''' Method to execute the training '''
  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    for Xs, Ys in train_loader:
      Xs = Xs.to(device=device)
      Ys = Ys.float().to(device=device)
      outputs = model(Xs)
      loss = loss_fn(outputs, Ys.view(-1,1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_train += loss.item()
    print(f'{datetime.datetime.now()} Epoch {epoch} \nTraining Loss {round(loss_train/len(train_loader),5)}')
    validate(model, val_loader)

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

model.load_state_dict(torch.load("<s", map_location=device))

#-------- Plot the predictions -----------------------------

plotPredictions(yActualTrain, yActualVal, yPredTrain, yPredVal)