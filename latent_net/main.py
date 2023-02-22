# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from Networks import LatentModel,TrainLatentNetwork
import os

save_path = os.getcwd()
latent_model = LatentModel()
train_latent = TrainLatentNetwork(latent_model)
train_latent.train_model()
train_latent.eval_mnist_latent(save_path)

