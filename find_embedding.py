import cPickle
import numpy as np
import theano
import theano.tensor as tensor
import nltk
#from model.text_gan import init_params, init_tparams
from model.cnn_layer import encoder
from model.utils import _p

import matplotlib
import matplotlib.pyplot as plt

from scipy import spatial
from collections import OrderedDict

theano.config.exception_verbosity='high'
def prepare_data_for_cnn(seqs_x, maxlen=40, filter_h=5, flag=True):
    
    lengths_x = [len(s) for s in seqs_x]
    
    if flag==True and maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None
    
    pad = filter_h -1
    x = []   
    for rev in seqs_x:    
        xx = []
        for i in xrange(pad):
            xx.append(0)
        for idx in rev:
            xx.append(idx)
        while len(xx) < maxlen + 2*pad:
            xx.append(0)
        x.append(xx)
    x = np.array(x,dtype='int32')
    return x

def prepare_data_for_rnn(seqs_x, maxlen=40):
    
    lengths_x = [len(s) for s in seqs_x]
    
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None
    
    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)

    x = np.zeros((maxlen_x, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen_x, n_samples)).astype(theano.config.floatX)
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx], idx] = 1.

    return x, x_mask

def find_sent_embedding(testset, params_file_path, n_words=22153, period = 887, img_w=300, img_h=148, feature_maps=300, 
    filter_hs=[3,4,5], n_x=300, n_h=500, n_h2 = 200, p_lambda_q = 0, p_lambda_fm = 1, n_codes = 2, max_epochs=16, lr_d=0.0001, lr_g = 0.00006, kde_sigma=1., batch_size=64, valid_batch_size=64,
    dim_mmd = 32, dispFreq=10, encodeDisFreq = 1, Large = 1e3, validFreq=5000, saveFreq=1000):
    data = np.load(params_file_path) 
    
    # n_h2 = data['dis_decoder_W1'].shape[0]
    options = {}

  
    options['n_words'] = n_words
    options['img_w'] = img_w
    options['img_h'] = img_h
    options['feature_maps'] = feature_maps
    options['filter_hs'] = filter_hs  #band width
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['n_h2'] = n_h2
    options['max_epochs'] = max_epochs
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['saveFreq'] = saveFreq
    options['encodeDisFreq'] = encodeDisFreq
    options['cnn_activation'] = 'tanh'

    # options['label_sizes'] = n_label
    
    # options['input_shape'] = (n_h2, n_dis*len(options['filter_hs']))
    # options['pred_shape'] = (n_label, n_h2)
    
    # options['dis_encoder_input_shape'] = (n_h2, options['feature_maps']*len(options['filter_hs'])+n_label)
   #  options['dis_encoder_pred_shape'] = (n_dis*len(options['filter_hs']), n_h2)
    
    # options['gan_input_shape'] = (n_h2, n_gan)
    # options['gan_pred_shape'] = (n_dis*len(options['filter_hs']), n_h2)
    # options['pred_d_shape'] = (2, n_h2)
    
    
    filter_w = img_w
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
        
    options['filter_shapes'] = filter_shapes
    options['pool_sizes'] = pool_sizes
    
    # params = init_params(options)
    tparams = OrderedDict()
    
    # tparams = init_tparams(data)
    
    for kk, pp in data.iteritems():
        tparams[kk] = theano.shared(data[kk], name=kk)
    
    
    # for kk, pp in params.iteritems():
    #     params[kk] = data[kk]
    #
    # for kk, pp in params.iteritems():
    #     tparams[kk].set_value(params[kk])

    x = tensor.matrix('x', dtype='int32')
    
    layer0_input = tparams['Wemb'][tensor.cast(x.flatten(),dtype='int32')].reshape((x.shape[0],1,x.shape[1],tparams['Wemb'].shape[1])) 
 
    layer1_inputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(tparams, layer0_input,filter_shape, pool_size, options,prefix=_p('cnn_d',i))                          
        layer1_input = conv_layer
        layer1_inputs.append(layer1_input)
    layer1_input = tensor.concatenate(layer1_inputs,1)
                                 
    f_embed = theano.function([x], [layer1_input] , name='f_embed')
    
    #print testset
    #x = prepare_data_for_cnn(testset)

    [sent_emb]  = f_embed(testset)
    
    #print str(testset.shape) + " " + str(x.shape) + " " + str(sent_emb.shape) + " " + str(sent_emb_dis.shape)
    return sent_emb
       
def predict_lab(z, params, prefix='dis_decoder'):
    
    """ z: size of (n_z, 1)
    """
    n_h2 = params[_p(prefix,'W1')].shape[0]
    n_label = params[_p(prefix,'V1')].shape[0]
    
        
    def sigmoid(x):
        return 1/(1+np.exp(-x))
        
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=1).reshape(x.shape[0],1)
    n_test = params['n_test']
        
    hidden_2_out = sigmoid(np.dot(z,params[_p(prefix,'W1')].T) + params[_p(prefix,'b1')])
    y_recons = sigmoid(np.dot(hidden_2_out,params[_p(prefix,'V1')].T) + params[_p(prefix,'c1')])
    y_recons = softmax(y_recons)
    print "y_recons shape:" + str(y_recons.shape)
        
    return y_recons     
       
       
       

       
       
def predict(z, params, beam_size, max_step, prefix='decoder'):
    
    """ z: size of (n_z, 1)
    """
    n_h = params[_p(prefix,'U')].shape[0]
    
    def _slice(_x, n, dim):
        return _x[n*dim:(n+1)*dim]
        
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    Vhid = np.dot(params['Vhid'],params['Wemb'].T)
    
    def _step(x_prev, h_prev, c_prev):
        preact = np.dot(h_prev, params[_p(prefix, 'U')]) + \
            np.dot(x_prev, params[_p(prefix, 'W')]) + \
            np.dot(z, params[_p(prefix, 'C')]) + params[_p(prefix, 'b')]
        
        i = sigmoid(_slice(preact, 0, n_h))
        f = sigmoid(_slice(preact, 1, n_h))
        o = sigmoid(_slice(preact, 2, n_h))
        c = np.tanh(_slice(preact, 3, n_h))
        
        c = f * c_prev + i * c
        h = o * np.tanh(c)
        
        y = np.dot(h, Vhid) + params['bhid']  
        
        return y, h, c
        
    h0 = np.tanh(np.dot(z, params[_p(prefix, 'C0')]) + params[_p(prefix, 'b0')])
    y0 = np.dot(h0, Vhid) + params['bhid']  
    c0 = np.zeros(h0.shape)
    
    maxy0 = np.amax(y0)
    e0 = np.exp(y0 - maxy0) # for numerical stability shift into good numerical range
    p0 = e0 / np.sum(e0)
    y0 = np.log(1e-20 + p0) # and back to log domain
    
    beams = []
    nsteps = 1
    # generate the first word
    top_indices = np.argsort(-y0)  # we do -y because we want decreasing order
    
    for i in xrange(beam_size):
        wordix = top_indices[i]
        # log probability, indices of words predicted in this beam so far, and the hidden and cell states
        beams.append((y0[wordix], [wordix], h0, c0))
    
    # perform BEAM search. 
    if beam_size > 1:
        # generate the rest n words
        while True:
            beam_candidates = []
            for b in beams:
                ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
                if ixprev == 0 and b[1]:
                    # this beam predicted end token. Keep in the candidates but don't expand it out any more
                    beam_candidates.append(b)
                    continue
                (y1, h1, c1) = _step(params['Wemb'][ixprev], b[2], b[3])
                y1 = y1.ravel() # make into 1D vector
                maxy1 = np.amax(y1)
                e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
                p1 = e1 / np.sum(e1)
                y1 = np.log(1e-20 + p1) # and back to log domain
                top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
                for i in xrange(beam_size):
                    wordix = top_indices[i]
                    beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1, c1)) #b0 : prob   b1: index
            beam_candidates.sort(reverse = True) # decreasing order
            beams = beam_candidates[:beam_size] # truncate to get new beams
            nsteps += 1
            if nsteps >= max_step: # bad things are probably happening, break out
                break
        # strip the intermediates
        predictions = [(b[0], b[1]) for b in beams]
    else:
        nsteps = 1
        h = h0
        c = c0
        # generate the first word
        top_indices = np.argsort(-y0)  # we do -y because we want decreasing order
        ixprev = top_indices[0]
        predix = [ixprev]
        predlogprob = y0[ixprev]
        #h_all = []
        while True:
            #h_all = [h] + h_all
            (y1, h, c) = _step(params['Wemb'][ixprev], h, c)
            ixprev, ixlogprob = ymax(y1)
            predix.append(ixprev)
            predlogprob += ixlogprob
            nsteps += 1
            if nsteps >= max_step:
                break
            predictions = [(predlogprob, predix)]

    return predictions
    
def ymax(y):
    """ simple helper function here that takes unnormalized logprobs """
    y1 = y.ravel() # make sure 1d
    maxy1 = np.amax(y1)
    e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
    p1 = e1 / np.sum(e1)
    y1 = np.log(1e-20 + p1) # guard against zero probabilities just in case
    ix = np.argmax(y1)
    return (ix, y1[ix]) 

def generate(z_emb, params_file_path, prefix = 'decoder_0'):
    params = np.load(params_file_path)
    beam_size = 1
    #print beam_size
    predset = []
    for i in xrange(len(z_emb)):
        pred = predict(z_emb[i], params, beam_size, max_step=40, prefix = prefix)
        predset.append(pred)
        #print i

    return predset

"""turn index matrix into real text"""
def matrix2text(predset):
    predset_text = []
    for sent in predset:
        rev = []
        for sen in sent:
            smal = []
            for w in sen[1]:
                if w != 0:
                    smal.append(ixtoword[w])
            rev.append(' '.join(smal))
        predset_text.append(rev)
    return predset_text

"""auxiliary function for KDE"""
def log_mean_exp(A,b,sigma):
    a=-0.5*((A-np.tile(b,[A.shape[0],1]))**2).sum(1)/(sigma**2)
    max_=a.max()
    return max_+np.log(np.exp(a-np.tile(max_,a.shape[0])).mean())

"""calculate negative KDE"""
def cal_nkde(X,mu):
    #print 'in KDE calculation'
    sigma = np.sqrt(np.median([((x-y)**2).sum() for x in X for y in mu]))
    E=sum([log_mean_exp(mu,x,sigma) for x in X])
    #E=0.0
    #for x in X:
    #    E=E+log_mean_exp(mu,x,sigma)
    Z=mu.shape[0]*np.log(sigma*np.sqrt(np.pi*2))
    return (Z-E)/(mu.shape[0]*X.shape[0])# Not sure about the denominator

""" BLEU score"""
def cal_BLEU(generated, reference):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    BLEUscore = 0.0
    cnt=0
    for g in generated:
        BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g)
    BLEUscore = BLEUscore/len(generated)
    return BLEUscore

if __name__ == '__main__':
    
    print "loading data..."
    x = cPickle.load(open("../../data/three_corpus.p","rb"))
    train, val, test                    = x[0], x[1], x[2]
    #train_text, val_text, test_text     = x[3], x[4], x[5]
    #train_y, val_y, test_y              = x[6], x[7], x[8]
    wordtoix, ixtoword                  = x[9], x[10]
    del x
    del train,val
    n_words = len(ixtoword)
    testset = prepare_data_for_cnn(test)
    #print test
    
    n_eval=10
    n_test=32
    n_gan=900
    kdes=np.zeros(10)
    bleus=np.zeros(10)
    autoencoder_d='autoencoder/disent_result_d.npz'
    autoencoder_g='autoencoder/disent_result_g.npz'
    mmd_d='mmd/disent_result_d.npz'
    mmd_g='mmd/disent_result_g.npz'
    print 'start'
    emb_test = find_sent_embedding(testset, autoencoder_d)
    for t in range(n_eval):
        z1 = np.random.uniform(-1,1,(n_test,n_gan)).astype('float32') 
        pred = generate(z1, mmd_g)
	predset=[[x for x in sent[0][1]] for sent in pred]
        predset = prepare_data_for_cnn(predset,flag=False)
	#print predset
	pred_text=matrix2text(pred)
	#print pred_text[0:9]
        emb_pred = find_sent_embedding(predset, autoencoder_d)
	kdes[t] = cal_nkde(emb_pred,emb_test)
	print 'Round '+ str(t) + ' KDE ' + str(kdes[t])
	bleus[t] = cal_BLEU([[str(x) for x in s] for s in predset], [[str(x) for x in s] for s in testset])
	print 'Round '+ str(t) + ' BLEU score ' + str(bleus[t])
    kde_std=np.std(kdes)
    kde_mean=np.mean(kdes)
    bleu_std=np.std(bleus)
    bleu_mean=np.mean(bleus)
    print 'Mean of KDEs in ' + str(n_eval) + ' experiment on textGAN' + str(kde_mean)
    print 'Standard Deviation of KDEs in ' + str(n_eval) + ' experiment on textGAN' + str(kde_std)
    print 'Mean of BLEU scores in ' + str(n_eval) + ' experiment on textGAN' + str(bleu_mean)
    print 'Standard Deviation of BLEU scores in ' + str(n_eval) + ' experiment on textGAN' + str(bleu_std)



#    whole = train + val + test
#    whole = test
    #whole_lab = train_y + val_y + test_y
#    whole_lab = test_y
#    min_lab = min(whole_lab)
#    whole_lab = [t- min_lab for t in whole_lab]
#    whole_text = train_text + val_text + test_text
#    del train, val, test
#    del train_text, val_text, test_text
    
#    print len(whole)
    
#    length = []
#    for sent in whole:
#        length.append(len(sent))
#    plt.hist(length)
    
    # the meaning of life is 6235, 13029, 18298, 868, 11429
    # my greatest accomplishment is 8430, 16792, 233, 11429
    # i could not live without 10149, 4457, 6955, 6752, 18137
#    n_test = 2000

#    if n_test > len(whole):
#         print "ERROR: choose a smaller number of test sample"
         
        
## question: all data? 
    # seed 22, 10000
#    np.random.seed(10000)
    
#    idx = np.arange(len(whole))
#    np.random.shuffle(idx)
    
#    testset = [whole[i] for i in idx[:n_test]]
    
#    (sent_emb, params, options) = find_sent_embedding(testset)
    
#    #z_emb = np.random.uniform(low=-1., high=1., size=(100, 512))
    
    # predict label and output
#    params['n_test'] = n_test
#    pred_label_prob = predict_lab(sent_emb_dis, params)  ## C
#    test_lab =  [whole_lab[i] for i in idx[:n_test]]
#    pred_label_prob = np.matrix(pred_label_prob)
#    avg_prob = pred_label_prob[range(n_test),test_lab].sum()/n_test
#    pred_acc = (np.argmax(pred_label_prob,axis=1).T == test_lab).astype(int).sum()/n_test
    
    
    ##
    
#    np.savez('params.npz',params=params,ixtoword=ixtoword,testset=testset,test_lab=test_lab,dis_emb=sent_emb_dis,sent_emb=sent_emb, options = options )

#     predset = generate(sent_emb, params)
#
#     testset_text = []
#     for sent in testset:
#         rev = []
#         for w in sent:
#             rev.append(ixtoword[w])
#         testset_text.append(' '.join(rev))
#
#     predset_text = []
#     for sent in predset:
#         rev = []
#         for sen in sent:
#             smal = []
#             for w in sen[1]:
#                 smal.append(ixtoword[w])
#             rev.append(' '.join(smal))
#         predset_text.append(rev)
#
#     #data = np.load('./result/bookcorpus_end.npz')
#     #history_negll = data['history_negll']
#     #plt.plot(history_negll[:,0])
#
#     for i in range(100):
#         print
#         print testset_text[i]
#         print predset_text[i][0]
#         print predset_text[i][1]
#         print predset_text[i][2]
#     #print predset_text[i][3]
#     #print predset_text[i][4]
#
#     """
#     johnny nodded his curly head , and then his breath eased into an even rhythm . 1.0
#  tam nodded his head , but his eyes rested on his mothers chin rather than her eyes . 0.414750796727
#
#  they probably use it for shows or meetings , kacey explained .
#     """
#
# #    pred = predict(sent_emb_ttt, params, beam_size=5, max_step=40)
# #
# #    rev = []
# #    for sen in pred:
# #        smal = []
# #        for w in sen[1]:
# #            smal.append(ixtoword[w])
# #        rev.append(' '.join(smal))
# #
#        
    
    
     
