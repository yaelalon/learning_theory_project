# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from biary_classifier_class import Trainer
import os

latent_df = pd.read_csv('mnist_latent.csv')
y = np.load('y.npy')
latent = latent_df.iloc[:,3:].to_numpy() 

loss_types = ['hinge','MSE']
gradient_types = ['GD','SGD','Constrained GD','regularized GD']
batch_size_dict = {'GD':8,'SGD':1,'Constrained GD':8,'regularized GD':8}
l_rate_vec = [0.001,0.005,0.01,0.05] 
results_df = pd.DataFrame(columns = ['loss type','gradient type','learning rate','Final loss','Final acc'])

count = -1

for l_rate in l_rate_vec:
    for loss_type in loss_types:
        for gradient_type in gradient_types:
            count +=1
            batch_size = batch_size_dict[gradient_type]
            trainer = Trainer(y,latent,loss_type = loss_type,gradient_type = gradient_type,batch_size = batch_size,learning_rate=l_rate)
            trainer.fit()
            results_df.loc[count,'loss type'] = loss_type
            results_df.loc[count,'gradient type'] = gradient_type
            results_df.loc[count,'Final loss'] = trainer.best_loss
            results_df.loc[count,'Final acc [%]'] = trainer.best_acc
            results_df.loc[count,'learning rate'] = l_rate

results_df.to_csv(os.path.join(os.path.dirname(os.getcwd()),'results.csv'))
