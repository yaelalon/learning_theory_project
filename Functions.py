# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:27:45 2023

@author: YaelAlon
"""

import numpy as np
import math

def mean_list(my_list):
    my_list = [item for item in my_list if not np.isnan(item)]
    if my_list == []:
        return math.nan
    else:
        return sum(my_list)/len(my_list)