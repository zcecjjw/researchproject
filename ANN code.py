#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:01:28 2020

@author: JonathanChew
"""

import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.datasets import load_iris
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, state_size, action_size):    #Here you define the weights of the layers
        super().__init__()
        self.layer1 = nn.Linear(state_size, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.action = nn.Linear(20, action_size)
        
    def get_weights(self):
        return self.weight
    
    def forward(self, state):                       #Here the layers are connected with activation functions
        m = torch.nn.LeakyReLU(0.1)#0.01)
        layer1 = m(self.layer1(state))
        layer2 = m(self.layer2(layer1))
        layer3 = m(self.layer3(layer2))
        action =(self.action(layer3))
        return (action)
    
x_m = 1000                                          #number of samples
x = np.random.default_rng().uniform(-5, 5, x_m)
y = np.random.default_rng().uniform(-5, 5, x_m)
X = np.array((x, y)).T
F = []

for i in range(0, x_m, 1):
    f = np.array([[100 * (y[i] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2, x[i]*y[i]]])
    F.append(*f.reshape(1,2))
F = np.array(F)

if f.ndim==1:
    F = F.reshape(-1, 1)
    
model = Model(X.shape[1],F.shape[1])                #initialising the neural network

# scale_x = preprocessing.normalize(X)    #To be completed
# scale_f = preprocessing.normalize(F)    #To be completed
scale_x = preprocessing.StandardScaler()
scale_f = preprocessing.StandardScaler()
X_scale = scale_x.fit_transform(X)
F_scale = scale_f.fit_transform(F)

#Split the set to train and test (validation)
X_train, X_test, F_train, F_test = train_test_split(X_scale,F_scale,test_size = 0.2, random_state=0)
# norm = preprocessing.MinMaxScaler()
# X_train_norm = norm.fit_transform(X_train)
# X_test_norm = norm.fit_transform(X_test)

class Data(Dataset):
    
    def __init__(self, X, F):
        X_dtype = torch.FloatTensor
        F_dtype = torch.FloatTensor
        self.length = X.shape[0]
        self.X_data = torch.from_numpy(X).type(X_dtype)
        self.F_data = torch.from_numpy(F).type(F_dtype)
        
    def __getitem__(self, index):
        return self.X_data[index], self.F_data[index]
    
    def __len__(self):
        return self.length
    
dataset_train = Data(X_train, F_train)
dataset_test = Data(X_test, F_test)
train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=len(dataset_test),shuffle=False)

def train(train_loader, model):
    """
    This function takes the batch data loader and performs for training for all epochs
    :param loader: All the data for each batch for X, F
    :param model: This is the neural net defined by you earlier
    :return: returns all the losses
    :rtype:  list with the losses
    """
    model.train()
    epoch = 100
    loss_fn = nn.MSELoss()  # Here we define the loss function (why MSE?)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.01)  # Here the optimizer is defined (Can you explain what it does?)
    losses = list()  # Initialize the losses
    batch_index = 0
    
    for e in range(epoch):
        for X, F in train_loader:
            
            F_predict = model(X)  # Forward propagation
            loss = loss_fn(F_predict, F)  # loss calculation
            optimizer.zero_grad()  # all grads of variables are set to 0 before backward calculation
            loss.backward()  # Backward propagation
            optimizer.step()  # update parameters
            loss_ = loss.data.item()
            batch_index += 1
            
        print("Epoch: ", e + 1, " Batches: ", batch_index, " Loss: ", loss_)
        losses.append(loss_)
        
    print('The loss of training is: ', losses[-1])
    
def perform_validation(model,test_loader):
    model.eval()
    loss_fn = nn.MSELoss()
    test_losses = list()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, F in test_loader:
            F_predict = model(X)
            test_loss = loss_fn(F_predict, F)
            test_loss_ = test_loss.data.item()
            accuracy  =sklearn.metrics.r2_score(F.data.numpy(),F_predict.data.numpy())
            
    print("Accuracy: ", accuracy)
    test_losses.append(test_loss_)
    
    print('The loss of validation is: ', test_losses[0])
    
train(train_loader,model)
perform_validation(model,test_loader)
