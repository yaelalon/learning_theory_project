# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:07:55 2023

@author: ShelleyJoyLevy
"""

import numpy as np
import pandas as pd


class BinaryClassifier:
    def __init__(self):
        self.weights = np.zeros((128,1)) # should we add a bias term? 

    @staticmethod    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod    
    def forward_pass(self,x):
        fc1 = np.dot(x, self.weights)
        y_pred = self.sigmoid(fc1)
        return y_pred

    
class Loss:
    def __int__(self,loss_type='MSE'):
        self.loss_type = loss_type
        
    
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
     def __init__(self, train=True):
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
    def __init__(self,loss_type='MSE',learning_rate=0.1, num_epochs=1000):
        self.model = BinaryClassifier()
        self.loss_cls = Loss()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.loss = [] # mean loss per epoch 
        self.acc = [] # mean acc per epoch 
        