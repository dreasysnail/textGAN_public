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

def param_init_mlp_layer(input_shape, pred_shape, params, prefix='mlp_layer'):
    
    """ input_shape: (num of hiddens, number of input features)
        pred_shape: (num of labels, number of hiddens)
    """
   
    W1 = np.asarray(rng.uniform(low=-0.01,high=0.01,size=input_shape),dtype=theano.config.floatX)
    b1 = np.ones((input_shape[0],), dtype=theano.config.floatX)*0.01   # initialize as 1 rather than 0
    V1 = np.asarray(rng.uniform(low=-0.01,high=0.01,size=pred_shape),dtype=theano.config.floatX) # 2*200
    c1 = np.ones((pred_shape[0],), dtype=theano.config.floatX)*0.01     # initialize as 1
    

      
    params[_p(prefix,'W1')] = W1
    params[_p(prefix,'b1')] = b1
    params[_p(prefix,'V1')] = V1
    params[_p(prefix,'c1')] = c1
    
    return params
    

def mlp_layer_softmax(tparams, layer1_input, prefix='mlp_layer'):
    
    """ layer1_input:  n_sample * n_feature    64*20
        input_shape: (num of hiddens, number of input features)   200*20
        pred_shape: (num of labels, number of hiddens) 2*200
        y_recon : n_label *n_sample 2*64
    """
    hidden_2_out = tensor.nnet.sigmoid(tensor.dot(layer1_input, tparams[_p(prefix,'W1')].T) + tparams[_p(prefix,'b1')] )  # 64*200  
    y_recons = tensor.dot(hidden_2_out, tparams[_p(prefix,'V1')].T) + tparams[_p(prefix,'c1')]  
    #y_recons = tensor.tanh(y_recons) * 10   # avoid numerical issues/label smoothing
    #y_recons = tensor.nnet.softmax(y_recons) # 64*2
    
    
    
    max_w = tensor.max(y_recons, axis = 1, keepdims=True)
    e0 = tensor.exp(y_recons - max_w)
    y_recons = e0 / tensor.sum(e0, axis = 1, keepdims=True)

    return y_recons
    
    
def mlp_layer_tanh(tparams, layer1_input, prefix='mlp_layer'):
    
    """ layer1_input:  n_sample * n_feature    64*20
        input_shape: (num of hiddens, number of input features)   200*20
        pred_shape: (num of labels, number of hiddens) 2*200
        y_recon : n_label *n_sample 2*64
    """
    hidden_2_out = tensor.nnet.sigmoid(tensor.dot(layer1_input, tparams[_p(prefix,'W1')].T) + tparams[_p(prefix,'b1')] )  # 64*200  
    y_recons = tensor.dot(hidden_2_out, tparams[_p(prefix,'V1')].T) + tparams[_p(prefix,'c1')]  
    #y_recons = tensor.tanh(y_recons) * 10   # avoid numerical issues/label smoothing
    #y_recons = tensor.nnet.softmax(y_recons) # 64*2
    
    y_recons = tensor.tanh(y_recons)

    return y_recons    
    
    
def mlp_layer_linear(tparams, layer1_input, prefix='mlp_layer'):
    
    """ layer1_input:  n_sample * n_feature    64*20
        input_shape: (num of hiddens, number of input features)   200*20
        pred_shape: (num of labels, number of hiddens) 2*200
        y_recon : n_label *n_sample 2*64
    """
    hidden_2_out = tensor.nnet.sigmoid(tensor.dot(layer1_input, tparams[_p(prefix,'W1')].T) + tparams[_p(prefix,'b1')] )  # 64*200  
    y_recons = tensor.dot(hidden_2_out, tparams[_p(prefix,'V1')].T) + tparams[_p(prefix,'c1')]  
    #y_recons = tensor.tanh(y_recons) * 10   # avoid numerical issues/label smoothing
    #y_recons = tensor.nnet.softmax(y_recons) # 64*2
    return y_recons      
    
    
    
def middle_layer(tparams, layer1_input, prefix='mlp_layer'):
    
    """ layer1_input:  n_sample * n_feature    64*20
        input_shape: (num of hiddens, number of input features)   200*20
        pred_shape: (num of labels, number of hiddens) 2*200
        y_recon : n_label *n_sample 2*64
    """
    hidden_2_out = tensor.nnet.sigmoid(tensor.dot(layer1_input, tparams[_p(prefix,'W1')].T) + tparams[_p(prefix,'b1')] )  # 64*200  
    # y_recons = tensor.dot(hidden_2_out, tparams[_p(prefix,'V1')].T) + tparams[_p(prefix,'c1')]  # avoid numerical issues
    # y_recons = tensor.nnet.softmax(y_recons) # 64*2

    return hidden_2_out
