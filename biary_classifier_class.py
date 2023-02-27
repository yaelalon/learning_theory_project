# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:07:55 2023

@author: ShelleyJoyLevy
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class BinaryClassifier:
    def __init__(self,gradient_type,loss_type):
        self.gradient_type = gradient_type
        self.weights = np.zeros((100,1)) # should we add a bias term? 
        self.loss_type = loss_type

    @staticmethod    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
       
    def forward_pass(self,x):
        fc1 = np.dot(x, self.weights)
        y_pred = self.sigmoid(fc1)
        return y_pred

    def loss(self,y,y_pred):
        if self.loss_type=='MSE':
            return np.mean((y_pred-y) ** 2)
        else:
            margin = np.maximum(0, 1 - y * y_pred) 
            return np.mean(margin) 
    
    def calc_loss_gradient(self,y,y_pred):
        if self.loss_type=='MSE':
            return 2*(y_pred-y)
        else:
            return 2
        
    def backward_pass(self,x,y,y_pred):
        dl = self.calc_loss_gradient(y,y_pred)
        da= self.forward_pass(x)*(1-self.forward_pass(x))
        dfc = x
        return dl*da*dfc
    
    def predict(self,x):
        y_pred = self.forward_pass(x)
        return 1*(y_pred>0.5)

class LatentMNIST_dataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        data_sample = torch.tensor(self.x[idx,:])
        label_sample = torch.tensor(self.y[idx])
        return data_sample, label_sample
    

class Trainer:
    def __init__(self,y,latent,loss_type='MSE',gradient_type = 'SGD',learning_rate=0.1, num_epochs=1000,batch_size=1,B=10):
        self.gradient_types = ['GD','Constrained GD','regularized GD','SGD']
        self.gradient_type = gradient_type
        if self.gradient_type not in self.gradient_types:
            raise('%s not it gradient type options' %(self.gradient_type))
            return
        
        self.loss_types = ['MSE','hinge']
        self.loss_type = loss_type
        if self.loss_type not in self.loss_types:
            raise('%s not it loss type options' %(self.loss_type))
            return
        self.x = latent
        self.seed = 0
        y[y<5] = 0
        y[y>=5] = 1
        self.y = y
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.20, random_state=self.seed)
        self.batch_size = batch_size
        self.model = BinaryClassifier(self.gradient_type,self.loss_type)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.loss = [] # mean loss per epoch 
        self.acc = [] # mean acc per epoch 
        self.training_dataloader = DataLoader(self.x_train,self.y_train, batch_size=1)
        self.test_dataloader = DataLoader(self.x_test,self.y_test, batch_size=1)
        self.model = BinaryClassifier(self.gradient_type,self.loss_type)
        self.B = B


    def proj_operator(w):
        B = np.eye(100)
        proj_v = np.zeros_like(w)
        for b in B.T:
            proj_v += np.dot(v, b) * b
        return proj_v

       
    def fit(self):
        for epoch in range(self.num_epochs):
            epoch_loss = []
            if self.gradient_type in ['GD','Constrained GD','regularized GD']:
                mean_gradient = []
                flag = True
            for batch in self.training_dataloader:
                x, y = batch
                y_pred = self.model.forward_pass(x)
                loss = self.model.loss(y,y_pred)
                epoch_loss.append(loss)
                gradient = self.model.backward_pass(x,y,y_pred)
                if flag:
                    mean_gradient.append(gradient)
                else:
                    self.model.weights -= gradient*self.learning_rate
            mean_gradient = np.mean(mean_gradient)
            epoch_loss = np.mean(epoch_loss)
            self.loss.appen(epoch_loss)
            print(f'Epoch Loss: {epoch_loss:.2f}')
            if self.gradient_type == 'GD':
                self.model.weights -= mean_gradient*self.learning_rate
            elif self.gradient_type == 'Constrained GD':
                temp_weights = self.model.weights.copy() - mean_gradient*self.learning_rate
                norm_v = np.linalg.norm(temp_weights)
                if norm_v <= self.B:
                    self.model.weights -= mean_gradient*self.learning_rate
                else:
                    self.model.weights = self.proj_operator(temp_weights)


            else:
                self.model.weights -= mean_gradient*self.learning_rate
            predictions = []
            for batch in self.test_dataloader:
                x, y = batch
                y_pred = self.model.predict(x)
                predictions.append(y_pred==y)
            acc = sum(predictions)/len(predictions)
            self.acc.append(acc)
            print(f'Epoch test accuracy: {acc:.2f}')


                
            
        
        
        
        