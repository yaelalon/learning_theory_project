# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:07:55 2023

@author: ShelleyJoyLevy
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

class BinaryClassifier:
    def __init__(self,gradient_type,loss_type):
        self.gradient_type = gradient_type
        self.weights = np.zeros((128,1)) # should we add a bias term? 
        self.loss_type = loss_type

    @staticmethod    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod    
    def forward_pass(self,x):
        fc1 = np.dot(x, self.weights)
        y_pred = self.sigmoid(fc1)
        return y_pred

    def loss(self,y,y_pred):
        if self.loss_type=='MSE':
            return np.mean((y - y_pred) ** 2)
        else:
            margin = np.maximum(0, 1 - y * y_pred) 
            return np.mean(margin) 
    
    def calc_loss_gradient(self,y,y_pred):
        if self.loss_type=='MSE':
            return 1
        else:
            return 2

class LatentMNIST_dataset():
    def __init__(self,csv_file, train=True):
       self.data = []
       self.labels = []
       with open(csv_file, 'r') as f:
           lines = f.readlines()
           for line in lines:
               line = line.strip().split(',')
               self.data.append(line[0])
               self.labels.append(line[1])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_sample = torch.tensor(self.data[idx])
        label_sample = torch.tensor(self.labels[idx])
        return data_sample, label_sample
    

class Trainer:
    def __init__(self,y,latent,loss_type='MSE',gradient_type = 'SGD',learning_rate=0.1, num_epochs=1000):
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

        self.model = BinaryClassifier(self.gradient_type,self.loss_type)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.loss = [] # mean loss per epoch 
        self.acc = [] # mean acc per epoch 
        