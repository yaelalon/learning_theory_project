# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:43:31 2023

@author: YaelAlon
"""

import numpy as np
import torch
from torchvision import datasets,transforms
import torch.nn as nn
import os
from Functions import mean_list
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math 
from sklearn.metrics import accuracy_score
import pandas as pd

class LatentModel(nn.Module):
    def __init__(self,f=1):
        super(LatentModel, self).__init__()
        self.num_latent_features = 100
        self.conv1 = nn.Conv2d(1, round(32*f), 3, 1)
        self.conv2 = nn.Conv2d(round(32*f), round(64*f), 3, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12*12*round(64*f), self.num_latent_features)
        self.fc2 = nn.Linear(self.num_latent_features, 10)     
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        output = self.fc2(x)
        output = F.softmax(output, dim=1)
        return output,x
                
class TrainLatentNetwork():    
    def __init__(self,latent_model):
        self.latent_model = latent_model
        self.batch_size = 64
        self.num_epochs = 10
        self.lrate = 0.0001
        self.folder_model = os.path.join(os.getcwd(),'latent_net')
        if not os.path.isdir(self.folder_model):
            os.mkdir(self.folder_model)
            
        self.load_data()

    def load_data(self):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))]) # Download and load the training data
        self.dataset_train = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=self.transform)
        self.Trainloader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True) # Download and load the test data
        self.dataset_dev = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=self.transform)
        self.DevLoader = torch.utils.data.DataLoader(self.dataset_dev, batch_size=self.batch_size, shuffle=True)
    
    
    def save_results_plot(self,train_history,dev_history,epoch,best_loss_epoch,measure_str,y_title):
        max_val = max(train_history + dev_history) + 0.5
        plt.figure(figsize=(8, 8))
        plt.title("Learning curve - " + measure_str, fontsize=16)
        plt.plot(train_history, label="Train " + measure_str)
        plt.plot(dev_history, label="Val " + measure_str)
        plt.plot(best_loss_epoch, dev_history[best_loss_epoch], marker="x", color="r", label="best model")
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel(y_title, fontsize=16)
        plt.ylim(0,max_val)
        plt.legend();
        plt.grid()
        plt.savefig(self.folder_model + '\\Learning curve - ' + measure_str + '.jpg')
    
    def calc_acc(self,y_label,output):
        output_label = torch.argmax(output,dim=1)
        acc = accuracy_score(output_label, y_label)
        return acc*100
        
    def train_model(self):
        print('Training latent network')
        acc = 0
        best_loss_epoch = math.inf
        lrate = self.lrate
        weight_decay = 0
        print_every = 30
        
        self.net = self.latent_model
        criterion = nn.CrossEntropyLoss()
            
        optimizer = torch.optim.Adam(self.net.parameters(),lr = lrate,weight_decay = weight_decay) #, weight_decay = weight_decay)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        Net_config = {'Learning_rate':self.lrate,'Batch_size':self.batch_size,'Num_epochs': \
                      self.num_epochs}
            
        #torch.save(Net_config, os.path.join(self.folder_model,'NetConfig.p'))
        train_loss_history = []
        train_acc_history = []

        dev_loss_history = []
        dev_acc_history = []

        
        for epoch in range(0,self.num_epochs):
            train_epoch_loss = []
            train_epoch_acc = []

            self.net.train()
            for i, data in enumerate(self.Trainloader,0):
                input_img, y_label = data
                output,_ = self.net(input_img)
                y = torch.zeros_like(output)
                for n in range(0,np.shape(y)[0]):
                    y[n,y_label[n]]=1
                
                optimizer.zero_grad()
                loss = criterion(y,output)

                loss.backward()
                optimizer.step()
                
                acc = self.calc_acc(y_label,output)
                
                num_images = (i+1)*self.batch_size
                total_num_images = len(self.Trainloader)*self.batch_size
                train_epoch_loss.append(loss.item())
                train_epoch_acc.append(acc)
                
                if i%print_every==0:
                    print("Training set - Epoch %d : %d\%d, loss = %.2f, acc = %.0f%s" %(epoch,num_images,total_num_images,train_epoch_loss[-1],train_epoch_acc[-1],'%'))

            train_loss_history.append(mean_list(train_epoch_loss))
            train_acc_history.append(mean_list(train_epoch_acc))            
            
            dev_epoch_loss = []
            dev_epoch_acc = []
            self.net.eval()
            for i, data in enumerate(self.DevLoader,0):
                input_img, y_label = data
                output,_ = self.net(input_img)
                y = torch.zeros_like(output)
                for n in range(0,np.shape(y)[0]):
                    y[n,y_label[n]]=1

                acc = self.calc_acc(y_label,output)

                loss  = criterion(y,output)                
                dev_epoch_loss.append(loss.item())
                dev_epoch_acc.append(acc)

            dev_loss = mean_list(dev_epoch_loss)
            dev_loss_history.append(dev_loss)        
            dev_acc_history.append(mean_list(dev_epoch_acc))
            
            if dev_loss<best_loss_epoch:
                best_loss_epoch = dev_loss
                best_loss_train = train_loss_history[-1]
                best_epoch = epoch
                self.best_acc = dev_acc_history[-1]
                #torch.save(net.state_dict(), self.folder_model + '\\' + 'BestDev.p')
            
            print("Validation set - Epoch %d : loss = %.2f, acc = %.0f%s" %(epoch,dev_loss,dev_acc_history[-1],'%'))
            self.save_results_plot(dev_loss_history,train_loss_history,epoch,best_epoch,'loss','loss')
            self.save_results_plot(dev_acc_history,train_acc_history,epoch,best_epoch,'acc','%')
            self.best_loss_epoch = best_loss_epoch
            self.best_loss_train = best_loss_train
            
    def eval_mnist_latent(self, save_path):
        batch_size = 256
        num_latent_features = self.net.num_latent_features
        x_train = self.dataset_train.train_data
        y_train = self.dataset_train.train_labels
        x_test = self.dataset_dev.train_data
        y_test = self.dataset_dev.train_labels

        x = np.concatenate((x_train,x_test))
        x = self.transform(x)
        x = np.swapaxes(x,0,1)
        x = np.swapaxes(x,1,2)
        
        y = np.concatenate((y_train,y_test))        
        self.net.eval()

        num_samples = np.shape(y)[0]
        iter = round(num_samples/batch_size)
        results_df = pd.DataFrame()
        out_label_all = []
        latent_all = np.zeros((0,num_latent_features))
        
        for n in range(0,iter):
            if n!=iter-1:
                ind = np.arange(n*batch_size,(n*batch_size+(batch_size))).tolist()
            else:
                ind = np.arange(n*batch_size,num_samples).tolist()
                
            input_image = x[ind,:,:]
            input_image = input_image[None, :]
            input_image = np.swapaxes(input_image,0,1)

            output,latent = self.net(input_image)
            latent = latent.detach().numpy()
            out_label = torch.argmax(output,dim=1).detach().numpy().tolist()
            out_label_all = out_label_all + out_label
            
            latent_all = np.concatenate((latent_all,latent))

            
        results_df['y'] = y
        results_df['y_predict'] = out_label_all
        for n in range(0,np.shape(latent_all)[1]):
            results_df[str(n)] = latent_all[:,n].tolist()
        
        results_df.to_csv('mnist_latent.csv')
        np.save('x.npy',x)
        np.save('y.npy',y)
        
        