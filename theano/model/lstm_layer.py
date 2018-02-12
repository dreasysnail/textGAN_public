
import numpy as np
import theano
import theano.tensor as tensor
from utils import _p, numpy_floatX
from utils import ortho_weight, uniform_weight, zero_bias

def param_init_encoder(options, params, prefix='encoder'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    
    W = np.concatenate([uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h)], axis=1)
    params[_p(prefix, 'W')] = W
    
    U = np.concatenate([ortho_weight(n_h),
                        ortho_weight(n_h),
                        ortho_weight(n_h),
                        ortho_weight(n_h)], axis=1)
    params[_p(prefix, 'U')] = U
    
    params[_p(prefix,'b')] = zero_bias(4*n_h)
    params[_p(prefix, 'b')][n_h:2*n_h] = 3*np.ones((n_h,)).astype(theano.config.floatX)

    return params
    
def encoder(tparams, state_below, mask, seq_output=False, prefix='encoder'):
    
    """ state_below: size of  n_steps * n_samples * n_x
    """

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1]

    n_h = tparams[_p(prefix,'U')].shape[0]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
                    tparams[_p(prefix, 'b')]

    def _step(m_, x_, h_, c_, U):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        f = tensor.nnet.sigmoid(_slice(preact, 1, n_h))
        o = tensor.nnet.sigmoid(_slice(preact, 2, n_h))
        c = tensor.tanh(_slice(preact, 3, n_h))
        
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    seqs = [mask, state_below_]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                    n_samples,n_h),
                                              tensor.alloc(numpy_floatX(0.),
                                                    n_samples,n_h)],
                                non_sequences = [tparams[_p(prefix, 'U')]],
                                name=_p(prefix, '_layers'),
                                n_steps=n_steps,
                                strict=True)
    
    h_rval = rval[0] 
    if seq_output:
        return h_rval
    else:
        # size of n_samples * n_h
        return h_rval[-1] 

def param_init_decoder(options, params, prefix='decoder'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    n_z = options['n_z']
    
    W = np.concatenate([uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h)], axis=1)
    params[_p(prefix, 'W')] = W
    
    U = np.concatenate([ortho_weight(n_h),
                        ortho_weight(n_h),
                        ortho_weight(n_h),
                        ortho_weight(n_h)], axis=1)
    params[_p(prefix, 'U')] = U
    
    C = np.concatenate([uniform_weight(n_z,n_h),
                        uniform_weight(n_z,n_h),
                        uniform_weight(n_z,n_h),
                        uniform_weight(n_z,n_h)], axis=1)
    params[_p(prefix,'C')] = C
    
    params[_p(prefix,'b')] = zero_bias(4*n_h)
    params[_p(prefix, 'b')][n_h:2*n_h] = 3*np.ones((n_h,)).astype(theano.config.floatX)

    
    C0 = uniform_weight(n_z, n_h)
    params[_p(prefix,'C0')] = C0
    
    params[_p(prefix,'b0')] = zero_bias(n_h)
    
    #params[_p(prefix,'b_y')] = zero_bias(n_x)  # 48
    
    return params   
    

def decoder(tparams, state_below, z, mask=None, prefix='decoder'):
    
    """ state_below: size of n_steps * n_samples * n_x 
        z: size of n_samples * n_z
    """

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1]
     
    n_h = tparams[_p(prefix,'U')].shape[0]

    #  n_samples * n_h
    state_belowx0 = tensor.dot(z, tparams[_p(prefix, 'C0')]) + \
            tparams[_p(prefix, 'b0')]
    h0 = tensor.tanh(state_belowx0)
    
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # n_steps * n_samples * n_h
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tensor.dot(z, tparams[_p(prefix, 'C')]) + tparams[_p(prefix, 'b')]
    
    def _step(m_, x_, h_, c_, U):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        f = tensor.nnet.sigmoid(_slice(preact, 1, n_h))
        o = tensor.nnet.sigmoid(_slice(preact, 2, n_h))
        c = tensor.tanh(_slice(preact, 3, n_h))
        
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
                          
    seqs = [mask[:n_steps-1], state_below_[:n_steps-1]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [h0,tensor.alloc(numpy_floatX(0.),
                                                    n_samples,n_h)],
                                non_sequences = [tparams[_p(prefix, 'U')]],
                                name=_p(prefix, '_layers'),
                                n_steps=n_steps-1,
                                strict=True)
                                
    h0x = tensor.shape_padleft(h0)
    h_rval = rval[0]
                            
    return tensor.concatenate((h0x,h_rval))  


    
    
    
# with thresholding    
def decoder_g(params,z, options, max_step, prefix='decoder'):
    
    """ z: size of (n_z, 1), use previous output as state below
    """
    n_h = params[_p(prefix,'U')].shape[0]
    
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]
    

    
    def _step(x_prev, h_prev, c_prev, z, U, W, C , b, Vhid, bhid, W_emb):        
        
        preact = tensor.dot(h_prev, U) + \
            tensor.dot(x_prev, W) + \
            tensor.dot(z, C) + b
        
        i = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        f = tensor.nnet.sigmoid(_slice(preact, 1, n_h))
        o = tensor.nnet.sigmoid(_slice(preact, 2, n_h))
        c = tensor.tanh(_slice(preact, 3, n_h))
        
        c = f * c_prev + i * c
        h = o * tensor.tanh(c)
        
        y = tensor.dot(h, Vhid) + bhid
        
        # not differentible
        # ixprev = tensor.argmax(y, axis=1)
        # y = W_emb[ixprev]
        if options['delta'] > 1e-10:
            y = tensor.switch(tensor.ge(y,options['delta']), y, 0)
        
        #pred_w = tensor.nnet.softmax(y*options['L'])  # simulate argmax
        maxy = tensor.max(y, axis= 1, keepdims=True)
        e0 = tensor.exp((y - maxy)*options['L']) # for numerical stability 
        pred_w = e0 / tensor.sum(e0, axis=1, keepdims=True)
        
        
        
        y = tensor.dot(pred_w, W_emb)
        
        
        return y, h, c
        
        
    # + params['bhid']  
    Vhid = tensor.dot(params['Vhid'],params['Wemb'].T)
        
    h0 = tensor.tanh(tensor.dot(z, params[_p(prefix, 'C0')]) + params[_p(prefix, 'b0')])
    y0 = tensor.dot(h0, Vhid) + params['bhid']
    c0 = tensor.zeros(h0.shape).astype(theano.config.floatX)
    

    if options['delta'] > 1e-10:
        y0 = tensor.switch(tensor.ge(y0,options['delta']), y0, 0)
    # pred_w = tensor.nnet.softmax(y0*options['L'])  # simulate argmax
    maxy0 = tensor.max(y0, axis=1, keepdims=True)
    e0 = tensor.exp((y0 - maxy0)*options['L']) # for numerical stability 
    pred_w = e0 / tensor.sum(e0, axis=1, keepdims=True)
    #y0 = tensor.log(1e-20 + p0) # and back to log domain
    
    y0 = tensor.dot(pred_w, params['Wemb'])


    
    rval, updates = theano.scan(_step,
                                outputs_info = [y0,h0,c0],
                                non_sequences = [z, params[_p(prefix, 'U')], params[_p(prefix, 'W')], params[_p(prefix, 'C')], params[_p(prefix, 'b')],Vhid,  params['bhid'], params['Wemb']],
                                name=_p(prefix, '_layers'),
                                n_steps=max_step-1,
                                strict=True)
    
    
    
    h0x = tensor.shape_padleft(h0)
    h_rval = rval[1]
                            
    return tensor.concatenate((h0x,h_rval))    
    
    
    
def decoder_emb_from_d(params, d_params,z, options, max_step, prefix='decoder'):
    
    """ z: size of (n_z, 1), use previous output as state below
    """
    n_h = params[_p(prefix,'U')].shape[0]
    
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]
    

    
    def _step(x_prev, h_prev, c_prev, U, W, C , b, Vhid, bhid, W_emb):        
        
        preact = tensor.dot(h_prev, U) + \
            tensor.dot(x_prev, W) + \
            tensor.dot(z, C) + b
        
        i = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        f = tensor.nnet.sigmoid(_slice(preact, 1, n_h))
        o = tensor.nnet.sigmoid(_slice(preact, 2, n_h))
        c = tensor.tanh(_slice(preact, 3, n_h))
        
        c = f * c_prev + i * c
        h = o * tensor.tanh(c)
        
        y = tensor.dot(h, Vhid) + bhid
        
        # not differentible
        # ixprev = tensor.argmax(y, axis=1)
        # y = W_emb[ixprev]
        y = tensor.switch(tensor.ge(y,options['delta']), y, 0)
        pred_w = tensor.nnet.softmax(y*options['L'])  # simulate argmax
        y = tensor.dot(pred_w, W_emb)
        
        
        return y, h, c
        
        
    # + params['bhid']  
    Vhid = tensor.dot(params['Vhid'],d_params['Wemb'].T)
        
    h0 = tensor.tanh(tensor.dot(z, params[_p(prefix, 'C0')]) + params[_p(prefix, 'b0')])
    y0 = tensor.dot(h0, Vhid) + params['bhid']
    c0 = tensor.zeros(h0.shape).astype(theano.config.floatX)
    
    # maxy0 = tensor.max(y0)
    # e0 = tensor.exp(y0 - maxy0) # for numerical stability shift into good numerical range
    # p0 = e0 / tensor.sum(e0)
    # y0 = tensor.log(1e-20 + p0) # and back to log domain
    y0 = tensor.switch(tensor.ge(y0,options['delta']), y0, 0)
    pred_w = tensor.nnet.softmax(y0*options['L'])  # simulate argmax
    y0 = tensor.dot(pred_w, d_params['Wemb'])


    
    rval, updates = theano.scan(_step,
                                outputs_info = [y0,h0,c0],
                                non_sequences = [params[_p(prefix, 'U')], params[_p(prefix, 'W')], params[_p(prefix, 'C')], params[_p(prefix, 'b')],Vhid,  params['bhid'], d_params['Wemb']],
                                name=_p(prefix, '_layers'),
                                n_steps=max_step-1,
                                strict=True)
    
    
    
    h0x = tensor.shape_padleft(h0)
    h_rval = rval[1]
                            
    return tensor.concatenate((h0x,h_rval))    
        
    
    
    # beams = []
#     nsteps = 1
#     # generate the first word
#     top_indices = np.argsort(-y0)  # we do -y because we want decreasing order
#
#     for i in xrange(beam_size):
#         wordix = top_indices[i]
#         # log probability, indices of words predicted in this beam so far, and the hidden and cell states
#         beams.append((y0[wordix], [wordix], h0, c0))
#
#     # perform BEAM search.
#     if beam_size > 1:
#         # generate the rest n words
#         while True:
#             beam_candidates = []
#             for b in beams:
#                 ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
#                 if ixprev == 0 and b[1]:
#                     # this beam predicted end token. Keep in the candidates but don't expand it out any more
#                     beam_candidates.append(b)
#                     continue
#                 (y1, h1, c1) = _step(params['Wemb'][ixprev], b[2], b[3])
#                 y1 = y1.ravel() # make into 1D vector
#                 maxy1 = np.amax(y1)
#                 e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
#                 p1 = e1 / np.sum(e1)
#                 y1 = np.log(1e-20 + p1) # and back to log domain
#                 top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
#                 for i in xrange(beam_size):
#                     wordix = top_indices[i]
#                     beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1, c1))
#             beam_candidates.sort(reverse = True) # decreasing order
#             beams = beam_candidates[:beam_size] # truncate to get new beams
#             nsteps += 1
#             if nsteps >= max_step: # bad things are probably happening, break out
#                 break
#         # strip the intermediates
#         predictions = [(b[0], b[1]) for b in beams]
#     else:
#         nsteps = 1
#         h = h0
#         # generate the first word
#         top_indices = np.argsort(-y0)  # we do -y because we want decreasing order
#         ixprev = top_indices[0]
#         predix = [ixprev]
#         predlogprob = y0[ixprev]
#         while True:
#             (y1, h) = _step(params['Wemb'][ixprev], h)
#             ixprev, ixlogprob = ymax(y1)
#             predix.append(ixprev)
#             predlogprob += ixlogprob
#             nsteps += 1
#             if nsteps >= max_step:
#                 break
#             predictions = [(predlogprob, predix)]
        
#    return predictions
