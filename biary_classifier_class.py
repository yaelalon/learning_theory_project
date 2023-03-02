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
import os
import matplotlib.pyplot as plt
import statistics
import math

class BinaryClassifier:
    def __init__(self,gradient_type,loss_type):
        self.gradient_type = gradient_type
        torch.manual_seed(100)
        self.weights = torch.randn(100, 1)
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
        gradient = (dl*da*dfc).T
        return gradient
    
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
    def __init__(self,y,latent,loss_type='MSE',gradient_type = 'SGD',learning_rate=0.0001, num_epochs=20,batch_size=1,B=10,lambda_val=0.001):
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
        
        self.save_path = os.path.join(os.path.dirname(os.getcwd()),'%s_%s' %(self.gradient_type,self.loss_type))
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
            
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
        
        self.training_data_set = LatentMNIST_dataset(self.x_train,self.y_train)
        self.training_dataloader = DataLoader(self.training_data_set,batch_size=self.batch_size,shuffle=True)
        self.test_data_set = LatentMNIST_dataset(self.x_test,self.y_test)        
        self.test_dataloader = DataLoader(self.test_data_set, batch_size=self.batch_size,shuffle=True)
        
        self.model = BinaryClassifier(self.gradient_type,self.loss_type)
        self.B = B
        self.lambda_val = lambda_val

    
    def save_results_plot(self,train_history,dev_history,best_loss_epoch,measure_str,y_title):
        max_val = max(train_history + dev_history) + 0.5
        
        if measure_str=='sensitivity' or measure_str=='specificity':
            max_val = 100
            
        plt.figure(figsize=(8, 8))
        plt.title("Learning curve - " + measure_str)
        plt.plot(train_history, label="Train " + measure_str)
        plt.plot(dev_history, label="Val " + measure_str)
        plt.plot(best_loss_epoch, dev_history[best_loss_epoch], marker="x", color="r", label="best model")
        plt.xlabel("Epochs")
        plt.ylabel(y_title)
        plt.ylim(0,max_val)
        plt.legend();
        plt.grid()
        plt.savefig(self.save_path + '\\Learning curve - ' + measure_str + '.jpg')
        
    def proj_operator(self,w):
        B = np.eye(100)
        proj_v = np.zeros_like(w)
        for b in B.T:
            b = np.expand_dims(b, axis=0)
            proj_v += np.dot(b,w.numpy())[0][0] * b.T
        return proj_v

    def weight_update(self):
        if self.gradient_type == 'GD':
            self.model.weights -= self.mean_gradient*self.learning_rate
            
        elif self.gradient_type == 'Constrained GD':
            temp_weights = self.model.weights.detach().clone() - self.mean_gradient*self.learning_rate
            norm_v = np.linalg.norm(temp_weights)
            if norm_v <= self.B:
                self.model.weights -= self.mean_gradient*self.learning_rate
            else:
                self.model.weights = self.proj_operator(temp_weights)
                
        elif self.gradient_type == 'regularized GD':
            self.model.weights -= self.learning_rate*(self.mean_gradient+self.lambda_val*self.model.weights)

    def mean_list(self,list):
        #list = [item.astype('float') for item in list if not np.isnan(item)]
        if list == []:
            return 0
        else:
            return round(statistics.mean(list),3)
    
    def calc_mean_gradient(self,gradient_list):
        gradient_array = np.stack(gradient_list, axis=0)
        gradient_array = gradient_array[:,:,0]
        mean_gradient = np.mean(gradient_array,axis = 0)
        mean_gradient = np.expand_dims(mean_gradient, axis=1)
        return mean_gradient
        
    def fit(self):
        print('Classification of mnist data set with %s and %s loss' %(self.gradient_type,self.loss_type))
        print_every = 10000
        flag = False
        self.train_loss = []
        self.train_acc = []
        self.dev_loss = []
        self.dev_acc = []
        best_loss_epoch = math.inf
        best_epoch = None
        total_samples = len(self.training_dataloader)*self.batch_size
        for epoch in range(self.num_epochs):
            train_epoch_loss = []
            train_epoch_acc = []
            if self.gradient_type in ['GD','Constrained GD','regularized GD']:
                self.gradient_list = []
                flag = True
            for num_batch,batch in enumerate(self.training_dataloader):
                x, y = batch
                y = y.numpy()
                y_pred = self.model.forward_pass(x)
                loss = self.model.loss(y,y_pred)
                train_epoch_loss.append(loss)
                acc = int(np.round(y_pred[0])==y)*100
                train_epoch_acc.append(acc)
                gradient = self.model.backward_pass(x.numpy(),y,y_pred)
                if flag:
                    self.gradient_list.append(gradient)
                else:
                    self.model.weights -= gradient*self.learning_rate
                
                if num_batch%print_every==0:
                    num_samples = (num_batch+1)*self.batch_size
                    print("Training set - Epoch %d : %d\%d, loss = %.2f, accuracy = %d%s" %(epoch,num_samples,total_samples,self.mean_list(train_epoch_loss),self.mean_list(train_epoch_acc),'%'))
                    
            self.train_loss.append(self.mean_list(train_epoch_loss))
            self.train_acc.append(self.mean_list(train_epoch_acc))
           
            if flag:
              self.mean_gradient = self.calc_mean_gradient(self.gradient_list)
              self.weight_update()
                
            self.dev_epoch_loss = []
            self.dev_epoch_acc = []
            for batch in self.test_dataloader:
                x, y = batch
                y = y.numpy()
                y_pred = self.model.predict(x)
                loss = self.model.loss(y,y_pred)
                self.dev_epoch_loss.append(loss)
                acc = int(np.round(y_pred[0])==y)*100
                self.dev_epoch_acc.append(acc)
                
            self.dev_loss.append(self.mean_list(self.dev_epoch_loss))
            self.dev_acc.append(self.mean_list(self.dev_epoch_acc))                
            print("Dev set - Epoch %d : loss = %.2f, accuracy = %d%s" %(epoch,self.mean_list(self.dev_epoch_loss),self.mean_list(self.dev_epoch_acc),'%'))

            if self.dev_loss[-1]<best_loss_epoch:
                best_loss_epoch = self.dev_loss[-1]
                best_epoch = epoch
                torch.save(self.model.weights, os.path.join(self.save_path,'best_weight.pt'))
            
            self.save_results_plot(self.train_acc,self.dev_acc,best_epoch,'Accuracy','Accuracy[%]')
            self.save_results_plot(self.train_loss,self.dev_loss,best_epoch,'Loss','Loss')
        
        print('\n')

                
            
        
        
        
        