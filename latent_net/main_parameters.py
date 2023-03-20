# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:44:14 2023

@author: yael
"""

from Networks import LatentModel,TrainLatentNetwork
import os

save_path = os.getcwd()
latent_model = LatentModel()
train_latent = TrainLatentNetwork(latent_model)
train_latent.train_model()
train_latent.eval_mnist_latent(save_path)

