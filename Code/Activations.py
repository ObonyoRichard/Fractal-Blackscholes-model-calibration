import torch.nn.functional as F
import matplotlib.pyplot as plt

def melu(x, α=0.49):
  ''' Custom ELU activation function which ensures C2 '''
  multiplier = (x>0)*((x/2 + 1 - 2*α)/(x + 1/α - 2)) + (x<0)*1
  return F.elu(x) * multiplier

def plotActivations():
  ''' Diagnostic method to plot melu '''
  x = torch.Tensor(np.arange(-2,2,0.1))
  y = melu(x)
  plt.plot(x.numpy(),y.numpy(),'r',label="MELU")
  y = F.gelu(x)
  plt.plot(x.numpy(),y.numpy(),'g',label="GELU")
  y = F.elu(x)
  plt.plot(x.numpy(),y.numpy(),'b',label="ELU")
  plt.legend(loc="upper left")

if __name__ == "__main__":
    plotActivations()