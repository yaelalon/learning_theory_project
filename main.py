# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from biary_classifier_class import Trainer
import os

def update_df_with_results(count,results_df,trainer):
    results_df.loc[count,'loss type'] = trainer.loss_type
    results_df.loc[count,'gradient type'] = trainer.gradient_type
    results_df.loc[count,'Final loss'] = trainer.best_loss
    results_df.loc[count,'Final acc [%]'] = trainer.best_acc
    results_df.loc[count,'learning rate'] = trainer.learning_rate                       
    return results_df

if __name__ =="__main__":     
    
    latent_df = pd.read_csv('mnist_latent.csv')
    latent = latent_df.iloc[:,3:].to_numpy() 
    
    loss_types = ['MSE','hinge']
    gradient_types = ['Constrained GD']#['regularized GD','GD','SGD','Constrained GD']
    batch_size_dict = {'GD':8,'SGD':1,'Constrained GD':8,'regularized GD':8}
    l_rate_vec = [0.001,0.005,0.01,0.05,0.1,0.5] 
    B_vec = [0.1,1,10]
    lambda_vec = [0.0001,0.001,0.001,0.01,0.1]
    results_df = pd.DataFrame(columns = ['loss type','gradient type','learning rate','Final loss','Final acc [%]'])
    count = -1   
    for loss_type in loss_types:
        for gradient_type in gradient_types:
            batch_size = batch_size_dict[gradient_type]
            for l_rate in l_rate_vec:
                if gradient_type == 'Constrained GD':
                    for B in B_vec:
                        y = np.load('y.npy')
                        count +=1
                        trainer = Trainer(y,latent,loss_type = loss_type,gradient_type = gradient_type,batch_size = batch_size,learning_rate=l_rate,B=B)
                        trainer.fit()
                        results_df = update_df_with_results(count,results_df,trainer)
                        results_df.loc[count,'B'] = B
                        
                elif gradient_type == 'regularized GD':
                    for lambda_val in lambda_vec:
                        y = np.load('y.npy')
                        count +=1
                        trainer = Trainer(y,latent,loss_type = loss_type,gradient_type = gradient_type,batch_size = batch_size,learning_rate=l_rate,lambda_val=lambda_val)
                        trainer.fit()
                        results_df = update_df_with_results(count,results_df,trainer)
                        results_df.loc[count,'lambda_val'] = lambda_val
                else:
                        count +=1
                        y = np.load('y.npy')
                        trainer = Trainer(y,latent,loss_type = loss_type,gradient_type = gradient_type,batch_size = batch_size,learning_rate=l_rate)
                        trainer.fit()
                        results_df = update_df_with_results(count,results_df,trainer)
                        
                del trainer
    
    results_df.to_csv(os.path.join(os.path.dirname(os.getcwd()),'results.csv'))
