
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


x_data = np.linspace(-10, 10, num=1000)
y_data = 0.1*x_data*np.cos(x_data) + 0.01*np.random.normal(size=1000)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

x_test = torch.tensor(x_test.reshape(-1,1),dtype = torch.float32)
y_test = torch.tensor(y_test.reshape(-1,1),dtype = torch.float32)


x_train = torch.tensor(x_train.reshape(-1,1),dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1,1),dtype=torch.float32)


class Net(torch.nn.Module):            
  def __init__(self):
    super().__init__()
    self.linear1 = torch.nn.Linear(1,32)
    self.linear2 = torch.nn.Linear(32,16)
    self.linear3 = torch.nn.Linear(16,8)
    self.linear4 = torch.nn.Linear(8,1)
    self.activation =torch.nn.ReLU()


  def forward(self,x):
    x=self.linear1(x)
    x = self.activation(x)
    x=self.linear2(x)
    x = self.activation(x)
    x=self.linear3(x)
    x = self.activation(x)
    x=self.linear4(x)
    return x       

model = Net()


loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_d = []


epochs = 1000
val_loss = []
for epoch in range(epochs):

  optimizer.zero_grad()
  #Prediction
  y_hat = model(x_train)

  #LOSS
  loss = loss_function(y_train,y_hat)
  loss_d.append(float(loss))
  #test
  y_test_hat = model(x_test)
  loss_test = loss_function(y_test,y_test_hat)
  val_loss.append(float(loss_test))
  #backpropagation ()
  loss.backward()



  #Update parameters
  optimizer.step() 
pred = model(x_train)
plt.scatter(x_train,pred.detach())
plt.scatter(x_train,y_train)
plt.show()