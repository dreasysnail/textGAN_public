import numpy as np
import theano
from theano import config
from collections import OrderedDict
import nltk
from pycocoevalcap.bleu.bleu import Bleu

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)
    
def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

     
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # if (minibatch_start != n):
    #     # Make a minibatch out of what is left
    #     minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)
    
def _p(pp, name):
    return '%s_%s' % (pp, name)

def dropout(X, trng, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

""" used for initialization of the parameters. """

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)
    
def uniform_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype(config.floatX)
    
def normal_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.randn(nin, nout) * scale
    return W.astype(config.floatX)
    
def zero_bias(ndim):
    b = np.zeros((ndim,))
    return b.astype(config.floatX)

"""auxiliary function for KDE"""
def log_mean_exp(A,b,sigma):
    a=-0.5*((A-theano.tensor.tile(b,[A.shape[0],1]))**2).sum(1)/(sigma**2)
    max_=a.max()
    return max_+theano.tensor.log(theano.tensor.exp(a-theano.tensor.tile(max_,a.shape[0])).mean())

'''calculate KDE'''
def cal_nkde(X,mu,sigma):
    s1,updates=theano.scan(lambda i,s: s+log_mean_exp(mu,X[i,:],sigma), sequences=[theano.tensor.arange(X.shape[0])],outputs_info=[np.asarray(0.,dtype="float32")])
    E=s1[-1]
    Z=mu.shape[0]*theano.tensor.log(sigma*np.sqrt(np.pi*2))
    return (Z-E)/mu.shape[0]


""" BLEU score"""
# def cal_BLEU(generated, reference):
#     #the maximum is bigram, so assign the weight into 2 half.
#     BLEUscore = 0.0
#     for g in generated:
#         BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g)
#     BLEUscore = BLEUscore/len(generated)
#     return BLEUscore

def cal_BLEU(generated, reference):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    BLEUscore = [0.0,0.0,0.0]
    for g in generated:
        score, scores = Bleu(4).compute_score(reference, {0: [g]})
        #print g, score
        for i, s in zip([0,1,2],score[1:]):
            BLEUscore[i]+=s
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    return BLEUscore
 
def prepare_for_bleu(sentence):
    sent=[x for x in sentence if x!=0]
    while len(sent)<4:
        sent.append(0)
    #sent = ' '.join([ixtoword[x] for x in sent])
    sent = ' '.join([str(x) for x in sent])
    return sent
    
