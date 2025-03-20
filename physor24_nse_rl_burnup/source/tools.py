# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:36:47 2023

@author: Majdi Radaideh
"""

import numpy as np

def calc_cumavg(data, N):
    """
    This function returns statistics every N data points of the vector data
    
    :param data: (list) vector of data points
    :param N: (scalar) size of a subgroup
    """
    cum_aves=[np.mean(data[i:i+N]) for i in range(0,len(data),N)]
    cum_std=[np.std(data[i:i+N]) for i in range(0,len(data),N)]
    cum_max=[np.max(data[i:i+N]) for i in range(0,len(data),N)]
    cum_min=[np.min(data[i:i+N]) for i in range(0,len(data),N)]

    return np.array(cum_aves), np.array(cum_std), np.array(cum_max), np.array(cum_min)
