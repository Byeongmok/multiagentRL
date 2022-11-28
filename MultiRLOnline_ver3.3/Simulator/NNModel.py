import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class NNModel(nn.Module):
    def __init__(self, NFEATURES, NHIDDEN):
        super(NNModel, self).__init__()
        self.NFEATURES = NFEATURES
        self.fc1 = nn.Linear(NFEATURES, NHIDDEN)
        self.fc2 = nn.Linear(NHIDDEN, NHIDDEN)
        self.fc3 = nn.Linear(NHIDDEN, NHIDDEN)
        self.fc4 = nn.Linear(NHIDDEN, NHIDDEN)
        self.fc5 = nn.Linear(NHIDDEN, 1)
        
    def forward(self, x):
        x = x.view(-1, self.NFEATURES)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = torch.sigmoid(x)
        x = x.view(-1)
        return x
    
    
class MLPModel():
    def __init__(self, NFEATURES, NHIDDEN, device):
        self.model = NNModel(NFEATURES, NHIDDEN).to(device)
        self.optimiser = Adam(self.model.parameters(), lr=0.002)
        self.device = device
    
    def predict(self, X):
        with torch.no_grad():
            # prepare data
            X = torch.tensor(X, dtype=torch.float).to(self.device)
            # forward pass
            output = self.model(X)
        return output[:,1].numpy()
    
    def fit(self, X, y):
        self.iterate_batch(X, y)
        return
    
    def iterate_batch(self, X, y):
        # prepare data
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)
        
        # zero gradients
        self.model.zero_grad()
        
        # forward pass
        output = self.model(X)
        
        # calculate loss
        loss = F.binary_cross_entropy(output, y.view(-1))
        
        # backward pass
        loss.backward()
        
        # update parameters
        self.optimiser.step()
        
        return output.detach().cpu().numpy()