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
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample



rng = np.random.RandomState(3435)

#def ReLU(x):
#    y = T.maximum(0.0, x)
#    return(y)

def param_init_batch_norm(input_shape,params, prefix='cnn'):

    """ input_shape: (num of hiddens, number of input features)
        pred_shape: (num of labels, number of hiddens)
    """

    beta = np.ones((input_shape[1],),dtype=theano.config.floatX) *0.01
    gamma = np.ones((input_shape[1],),dtype=theano.config.floatX) *0.1

    params[_p(prefix,'beta')] = beta
    params[_p(prefix,'gamma')] = gamma

    return params


def batch_norm(tparams, input, options, prefix='cnn'):

    """ layer1_input:  n_sample * n_feature    64*20
        input_shape: (num of hiddens, number of input features)   200*20
        pred_shape: (num of labels, number of hiddens) 2*200
        y_recon : n_label *n_sample 2*64
    """

    input_hat=(input-input.mean(0))/(input.std(0)+1.0/options['L'])
    input_=input_hat*tparams[_p(prefix,'gamma')]+tparams[_p(prefix,'beta')]
    return input_
