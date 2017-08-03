# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:30:58 2016

@author: yizhe
"""
import numpy as np
import theano
import theano.tensor as tensor
import theano.tensor.shared_randomstreams
from utils import _p




rng = np.random.RandomState(3435)

#def ReLU(x):
#    y = T.maximum(0.0, x)
#    return(y)

def param_init_ss(input_size, params):
    
    """ input_shape: (num of hiddens, number of input features)
        pred_shape: (num of labels, number of hiddens)
    """   

    params['acc_fake_xx'] = np.eye(input_size)*1
    params['acc_real_xx'] = np.eye(input_size)*1
    params['acc_fake_mean'] = np.zeros((input_size))
    params['acc_real_mean'] = np.zeros((input_size))
    params['seen_size'] = 0

    return params
    

