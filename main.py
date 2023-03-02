# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from biary_classifier_class import Trainer

latent_df = pd.read_csv('mnist_latent.csv')
#x = np.load('x.npy')
y = np.load('y.npy')
latent = latent_df.iloc[:,3:].to_numpy() 

gradient_types = ['Constrained GD','GD','SGD','regularized GD']
for gradient_type in gradient_types:
    trainer = Trainer(y,latent,loss_type='MSE',gradient_type = gradient_type)
    trainer.fit()
