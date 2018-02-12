
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

def param_init_encoder(filter_shape, params, prefix='cnn_d'):
    
    """ filter_shape: (number of filters, num input feature maps, filter height,
                        filter width)
        image_shape: (batch_size, num input feature maps, image height, image width)
    """
   
    W = np.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape),dtype=theano.config.floatX)
    b = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
      
    params[_p(prefix,'W')] = W
    params[_p(prefix,'b')] = b

    return params
    

def encoder(tparams, layer0_input, filter_shape, pool_size, options, prefix='cnn_d'):
    
    """ filter_shape: (number of filters, num input feature maps, filter height,
                        filter width)
        image_shape: (batch_size, num input feature maps, image height, image width)
    """
    
    conv_out = conv.conv2d(input=layer0_input, filters=tparams[_p(prefix,'W')], filter_shape=filter_shape)
    # conv_out_tanh = tensor.tanh(conv_out + tparams[_p(prefix,'b')].dimshuffle('x', 0, 'x', 'x'))
    # output = downsample.max_pool_2d(input=conv_out_tanh, ds=pool_size, ignore_border=False)
    
    if options['cnn_activation'] == 'tanh':
        conv_out_tanh = tensor.tanh(conv_out + tparams[_p(prefix,'b')].dimshuffle('x', 0, 'x', 'x'))
        output = downsample.max_pool_2d(input=conv_out_tanh, ds=pool_size, ignore_border=False)  # the ignore border is very important
    elif options['cnn_activation'] == 'linear':
        conv_out2 = conv_out + tparams[_p(prefix,'b')].dimshuffle('x', 0, 'x', 'x')
        output = downsample.max_pool_2d(input=conv_out2, ds=pool_size, ignore_border=False)  # the ignore border is very important
    else:
        print(' Wrong specification of activation function in CNN')

    return output.flatten(2)
    
    #output.flatten(2)
