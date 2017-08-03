#import os
import sys
import time
import logging
import cPickle

import numpy as np
import theano
import theano.tensor as tensor

from model.text_gan import init_params, init_tparams, build_model
from model.optimizers import Adam, RMSprop_v1
from model.utils import get_minibatches_idx, zipp, unzip, cal_BLEU, prepare_for_bleu
from collections import OrderedDict
from foxhound.theano_utils import floatX, sharedX
from foxhound import updates as update_grad

sys.setrecursionlimit(10000)
#theano.config.optimizer='fast_compile'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

def prepare_data_for_cnn(seqs_x, maxlen=60, filter_h=5):
    
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
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

def prepare_data_for_rnn(seqs_x, maxlen=60):
    
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
    
def calu_cost(f_cost, prepare_data_for_cnn, prepare_data_for_rnn, val, val_lab, kf):
    # validation cost
    total_negll = 0.
    total_lab = 0.
    total_cr = 0.
    total_cf = 0.
    total_len = 0.
    
    for _, train_index in kf:
        sents = [val[t] for t in train_index]
                
        x = prepare_data_for_cnn(sents)
        y, y_mask = prepare_data_for_rnn(sents)
        ## modified
        lab = [val_lab[t] for t in train_index]
        ##
        negll, cost_lab, cost_cr, cost_cf = f_cost(x, y, y_mask, lab) 
        negll, cost_lab, cost_cr, cost_cf = negll * np.sum(y_mask), cost_lab* np.sum(y_mask), cost_cr * np.sum(y_mask), cost_cf * np.sum(y_mask)
        length = np.sum(y_mask)
        total_negll += negll
        total_lab += cost_lab
        total_cr += cost_cr
        total_cf += cost_cf
        total_len += length

    return total_negll/total_len, total_lab/total_len, total_cr/total_len, total_cf/total_len
    
    
def calu_cost_gan(f_cost, prepare_data_for_cnn, prepare_data_for_rnn, val, val_z , kf):
    # validation cost
    total_negll = 0.
    total_gan = 0.
    total_len = 0.
    total_KDE = 0.
    total_KDE_i = 0.
        
    for _, train_index in kf:
        sents = [val[t] for t in train_index]
                
        x = prepare_data_for_cnn(sents)
        #y, y_mask = prepare_data_for_rnn(sents)
        ## modified
#        lab = [val_lab[t] for t in train_index]
        ##
        z = [val_z[t] for t in train_index]
        negll, cost_gan, KDE_val, KDE_val_i = f_cost(x, z) 
        # negll, cost_gan = negll * np.sum(y_mask), cost_gan * np.sum(y_mask)
        # length = np.sum(y_mask)
        total_negll += negll
        total_gan += cost_gan
        total_KDE += KDE_val
        total_KDE_i += KDE_val_i
        total_len += 1

    return total_negll/total_len, total_gan/total_len, total_KDE/total_len, total_KDE_i/total_len

""" Training the model. """
# original lr: 0.001
def train_model(train, val, test, train_lab, val_lab, test_lab, ixtoword, n_words=22153, period = 887, img_w=300, img_h=148, feature_maps=300, 
    filter_hs=[3,4,5], n_x=300, n_h=500,  n_h2_d = 200 , n_h2 = 900, p_lambda_q = 0, p_lambda_fm = 0.001, p_lambda_recon = 0.001, n_codes = 2, max_epochs=16, lr_d=0.0001, lr_g = 0.00005, kde_sigma=1., batch_size=256, valid_batch_size=256,
    dim_mmd = 32, dispFreq=10, dg_ratio = 1, Large = 1e3, validFreq=500, saveFreq=500, saveto = 'disent_result'):
        
    """ n_words : word vocabulary size
        feature_maps : CNN embedding dimension for each width
        filter_hs : CNN width
        n_h : LSTM/GRU number of hidden units 
        n_h2: discriminative network number of hidden units
        n_gan: number of hidden units in GAN
        n_codes: number of latent codes 
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
    """
    n_gan = len(filter_hs)*feature_maps  # 900
    
    options = {}
    options['n_words'] = n_words
    options['img_w'] = img_w
    options['img_h'] = img_h
    options['feature_maps'] = feature_maps
    options['filter_hs'] = filter_hs  #band width
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['n_h2'] = n_h2
    options['n_h2_d'] = n_h2_d
    options['n_codes'] = n_codes
    options['lambda_q'] = p_lambda_q
    options['lambda_fm'] = p_lambda_fm  # weight for feature matching
    options['lambda_recon'] = p_lambda_recon
    options['L'] = Large
    options['max_epochs'] = max_epochs
    options['lr_d'] = lr_d
    options['lr_g'] = lr_g 
    options['kde_sigma']=kde_sigma
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['saveFreq'] = saveFreq
    options['dg_ratio'] = dg_ratio

    options['n_gan'] = n_gan 
    options['debug'] = False
    options['feature_match'] = 'mmd' #'mmd' #' mmd_h','mmd_ld' #'JSD_acc'  # moment  #None  # 
    options['shareLSTM'] = True
    options['delta'] = 0.00
    options['sharedEmb'] = False
    options['cnn_activation'] = 'tanh' # tanh
    options['sigma_range'] = [20] # range of sigma for mmd
    options['diag'] = 0.1 # diagonal matrix added on cov for JSD_acc
    options['label_smoothing'] = 0.01
    options['dim_mmd'] = dim_mmd
    options['force_cut'] = 'None'
    options['batch_norm'] = False
    options['wgan'] = False
    options['cutoff'] = 0.01
    
    
    options['max_step'] = 60
    options['period'] = period
   
    logger.info('Model options {}'.format(options))

    logger.info('Building model...')
    
    filter_w = img_w
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
        
    options['filter_shapes'] = filter_shapes
    options['pool_sizes'] = pool_sizes
    # generative model for GAN 
    
    
    ## modified
    n_label = len(set(train_lab))
    options['label_sizes'] = n_label
    
    n_feature = len(options['filter_hs'])*options['feature_maps']

    options['input_shape'] = (n_h2_d, n_feature)  
    options['pred_shape'] = (1, n_h2_d)
    
    if options['feature_match'] == 'mmd_ld':
        options['mmd_shape'] = (dim_mmd, n_h2_d) 
    
    options['propose_shape'] = (n_codes, n_h2_d) 
    
    
    # if options['reverse']:
    options['input_recon_shape'] = (n_h2, n_feature)  
    options['recon_shape'] = (n_gan, n_h2) if options['shareLSTM'] else (n_gan+1, n_h2)
    ##
    
    d_params_s, g_params_s, s_params_s = init_params(options)
    d_params, g_params, s_params = init_tparams(d_params_s, g_params_s, s_params_s, options)
    lr_d_t = tensor.scalar(name='lr_d')
    lr_g_t = tensor.scalar(name='lr_g')
    
    use_noise, use_noise2, x, z, d_cost, g_cost,r_cost, fake_recon, acc_fake_xx, acc_real_xx, acc_fake_mean, acc_real_mean, wtf1, wtf2, wtf3, wtf4, wtf5, wtf6, KDE, KDE_input = build_model(d_params, g_params, s_params, options)  # change
    f_cost = theano.function([x, z], [d_cost, g_cost, KDE, KDE_input], name='f_cost')
    #f_print = theano.function([x, z],[ wtf1, wtf2, wtf3, wtf4, wtf5, wtf6, KDE, KDE_input], name='f_print',on_unused_input='ignore')
    f_print = theano.function([x, z],[ wtf1, wtf2, wtf3, wtf4, wtf5, wtf6], name='f_print')
    f_recon = theano.function([x, z],[ r_cost, fake_recon, d_cost], name='f_recon',on_unused_input='ignore')
    
    if options['feature_match']:
        ss_updates =[(s_params['acc_fake_xx'], acc_fake_xx),
                    (s_params['acc_real_xx'], acc_real_xx),
                    (s_params['acc_fake_mean'], acc_fake_mean),
                    (s_params['acc_real_mean'], acc_real_mean),
                    (s_params['seen_size'],s_params['seen_size']+options['batch_size'])]
        f_update_ss = theano.function([x, z], s_params , updates= ss_updates)
    
   
    f_cost_d, _train_d = Adam(d_params, d_cost, [x, z], lr_d_t)
    if options['feature_match']:
        f_cost_g, _train_g = Adam(g_params, g_cost, [x, z], lr_g_t)
    else:
        f_cost_g, _train_g = Adam(g_params, g_cost, [z], lr_g_t)


    ## 
    

    
    logger.info('Training model...')
    
    history_cost = []  
    uidx = 0  # the number of update done
    kdes = np.zeros(10)
    kde_std = 0. # standard deviation of every 10 kde_input
    kde_mean = 0.

    start_time = time.time()
    
    kf_valid = get_minibatches_idx(len(val), valid_batch_size)
    y_min = min(train_lab)
    train_lab = [t - y_min for t in train_lab]
    val_lab = [t - y_min for t in val_lab]
    test_lab = [t - y_min for t in test_lab]
    testset = [prepare_for_bleu(s) for s in test[:1000]]
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0
            
            kf = get_minibatches_idx(len(train), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(0.0)
                use_noise2.set_value(0.0)
                sents = [train[t] for t in train_index]
                
                x = prepare_data_for_cnn(sents)
                n_samples += x.shape[0]
                
                if options['shareLSTM']:
                    z = np.random.uniform(-1,1,(batch_size, n_gan)).astype('float32')
                else:
                    z = np.random.uniform(-1,1,(batch_size, n_gan+1)).astype('float32')       
                
                z[:,0] = np.random.randint(n_codes, size = batch_size).astype('float32')                                      
                
		        # update gradient
                if options['feature_match']:
                    cost_g = f_cost_g(x,z)
                else:
                    cost_g = f_cost_g(z)
                    
                if np.isnan(cost_g):
                    
                    logger.info('NaN detected')
                    temp_out = f_print(x,z)
                    
                    print 'real'+str(temp_out[0]) + ' fake' + str(temp_out[1])
                    return 1., 1., 1.
                    
                if np.isinf(cost_g):
                    temp_out = f_print(x,z)
                    print 'real'+str(temp_out[0]) + ' fake' + str(temp_out[1])
                    logger.info('Inf detected')
                    return 1., 1., 1.
                    
                # update G  
                _train_g(lr_g)

                if np.mod(uidx, dispFreq) == 0:
                    temp_out = f_print(x,z)
                    _, _, cost_d = f_recon(x,z)
                    
                    np.set_printoptions(precision=3)
                    np.set_printoptions(threshold=np.inf)
       
                    print 'real '+str(round(temp_out[0],2)) + ' fake ' + str(round(temp_out[1],2)) + ' Covariance loss ' + str(round(temp_out[3],2)) +' mean loss '+ str(round(temp_out[5],2))
                    print 'cost_g ' + str(cost_g) +  ' cost_d ' + str(cost_d)
                    print("Generated:" + " ".join([ixtoword[x] for x in temp_out[2][0] if x != 0]))
                    
                    logger.info('Epoch {} Update {} Cost G {} Real {} Fake {} loss_cov {}  meanMSE {}'.format(eidx, uidx, cost_g, round(temp_out[0],2), round(temp_out[1],2), temp_out[3],temp_out[5])) 
                    logger.info('Generated: {}'.format(" ".join([ixtoword[x] for x in temp_out[2][0] if x != 0]))) 
                    
                if np.mod(uidx, dg_ratio) == 0:
                    x = prepare_data_for_cnn(sents) 
                    cost_d = f_cost_d(x, z)
                    _train_d(lr_d)
                                                
                    if np.mod(uidx, dispFreq) == 0:
                        logger.info('Cost D {}'.format(cost_d))
                
                if np.mod(uidx, saveFreq) == 0:
                    
                    logger.info('Saving ...')
                    
                    d_params_s = unzip(d_params)
                    g_params_s = unzip(g_params)
                    params_d = OrderedDict()
                    params_g = OrderedDict()
                    for kk, pp in d_params_s.iteritems():
                        params_d[kk] = np.asarray(d_params_s[kk])
                    for kk, pp in g_params_s.iteritems():
                        params_g[kk] = np.asarray(g_params_s[kk])
                    
                    np.savez(saveto + '_d.npz', history_cost=history_cost, options = options, **params_d)
                    np.savez(saveto + '_g.npz', history_cost=history_cost, options = options, **params_g)
                    
                    logger.info('Done ...')

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    use_noise2.set_value(0.)
                    if options['shareLSTM']:
                        val_z = np.random.uniform(-1,1,(batch_size,n_gan)).astype('float32')
                    else:
                        val_z = np.random.uniform(-1,1,(batch_size,n_gan+1)).astype('float32')
                    
                    temp_out = f_print(x,val_z)
                    predset = temp_out[2]
                    [bleu2s,bleu3s,bleu4s] = cal_BLEU([prepare_for_bleu(s) for s in predset], {0: testset})
                        
                    logger.info('Valid BLEU2 = {}, BLEU3 = {}, BLEU4 = {}'.format(bleu2s,bleu3s,bleu4s))
                    print 'Valid BLEU (2,3,4): ' + ' '.join([str(round(it,3)) for it in (bleu2s,bleu3s,bleu4s)])
                
                    
                if options['feature_match']:
                    f_update_ss(x,z)

        logger.info('Seen {} samples'.format(n_samples))

    except KeyboardInterrupt:
        logger.info('Training interrupted')

    end_time = time.time()
    
    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    
    return valid_cost

if __name__ == '__main__':
    
    # create logger with 'eval_disentangle'
    # https://docs.python.org/2/howto/logging-cookbook.html
    logger = logging.getLogger('eval_textGAN')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_textGAN.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    
    x = cPickle.load(open("../../data/data.p","rb"))
    train, val, test                    = x[0], x[1], x[2]
    train_text, val_text, test_text     = x[3], x[4], x[5]
    train_lab, val_lab, test_lab        = x[6], x[7], x[8]
    wordtoix, ixtoword                  = x[9], x[10]
    del x
    del train_text, val_text, test_text
    
    period = wordtoix['.']
    
    n_words = len(ixtoword)
    
    valid_cost = train_model(train, val, test, train_lab, val_lab, test_lab, ixtoword, n_words=n_words, period = period)
