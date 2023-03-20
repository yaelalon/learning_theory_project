# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


    
from Networks import LatentModel,TrainLatentNetwork
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_results_plot(train_history,dev_history,measure_str,factor_vec):
    max_val = max(train_history + dev_history) + 0.5
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve - " + measure_str, fontsize=16)
    plt.plot(factor_vec,train_history, label="Train " + measure_str)
    plt.plot(factor_vec,dev_history, label="Val " + measure_str)
    plt.xlabel("Factor", fontsize=16)
    plt.ylabel('loss', fontsize=16) 
    plt.ylim(0,max_val)
    plt.legend();
    plt.grid()


param_foctor_vec = np.arange(0.1,3,0.1)
results_df = pd.DataFrame(columns= ['Factor','best loss dev','best loss train'])

for n,f in enumerate(param_foctor_vec):
    print('\nTraining net with factor %.1f' %(f))
    save_path = os.getcwd()
    latent_model = LatentModel(f)
    train_latent = TrainLatentNetwork(latent_model)
    train_latent.train_model()
    train_latent.eval_mnist_latent(save_path)
    results_df.loc[n,'Factor'] = f
    results_df.loc[n,'best loss dev'] = train_latent.best_loss_epoch
    results_df.loc[n,'best loss train'] = train_latent.best_loss_train
    results_df.to_csv('latent_comparison.csv')
    
save_results_plot(results_df['best loss train'],results_df['best loss dev'],'loss',results_df['Factor'])