
import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import _p
from utils import uniform_weight, zero_bias
from utils import cal_nkde

from cnn_layer import param_init_encoder, encoder
from lstm_layer import param_init_decoder, decoder, decoder_g
from mlp_layer import param_init_mlp_layer, mlp_layer_softmax, mlp_layer_tanh, middle_layer, mlp_layer_linear
from ss import param_init_ss
from batch_norm import param_init_batch_norm, batch_norm


# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  
def init_params(options):
    
    n_words = options['n_words']
    n_x = options['n_x']  
    n_h = options['n_h']   # number of hidden units in D
    
    d_params = OrderedDict()
    g_params = OrderedDict()
    s_params = OrderedDict()
    # word embedding 
    emb_params = uniform_weight(n_words,n_x)
    length = len(options['filter_shapes'])
    
    d_params['Wemb'] = emb_params  # (d)
    g_params['Wemb'] = emb_params
    s_params = param_init_ss(length*options['feature_maps'], s_params)
    #params['Wemb'] = W.astype(config.floatX)
    # encoding words into sentences (cnn, d)

    for idx in xrange(length):
        d_params = param_init_encoder(options['filter_shapes'][idx],d_params,prefix=_p('cnn_d',idx))
        g_params = param_init_encoder(options['filter_shapes'][idx],g_params,prefix=_p('cnn_d',idx))
    
    options['n_z'] = options['feature_maps'] * length

    # discriminative (d)
    d_params = param_init_mlp_layer(options['input_shape'],options['pred_shape'],d_params,prefix='dis_d') 
    
    if options['feature_match'] == 'mmd_ld':
        d_params = param_init_mlp_layer(options['input_shape'],options['mmd_shape'],d_params,prefix='dis_mmd') 
    
    # discriminative (q)
    g_params = param_init_mlp_layer(options['input_shape'],options['propose_shape'],g_params,prefix='dis_q')
    
    #if options['reverse']:
    d_params = param_init_mlp_layer(options['input_recon_shape'],options['recon_shape'],d_params,prefix='recon') 
    
    # batch norm
    
    if options['batch_norm']:
        d_params = param_init_batch_norm(options['input_shape'],d_params,prefix='real')  
        d_params = param_init_batch_norm(options['input_shape'],d_params,prefix='fake')     
    
 
    # lstm (gen)
    if options['shareLSTM']:
        g_params = param_init_decoder(options,g_params,prefix='decoder_0')
    else:
        for idx in range(options['n_codes']):  # don't use xrange
            g_params = param_init_decoder(options,g_params,prefix=_p('decoder',idx))
   
       
    g_params['Vhid'] = uniform_weight(n_h,n_x)
    g_params['bhid'] = zero_bias(n_words)                                    
    try: 
        data = np.load('./disent_result_g.npz') 
        print('Use saved initialization: disent_result_g.npz ...')
        for kk, pp in g_params.iteritems():
            if kk in data.keys() and g_params[kk].shape == data[kk].shape :
                print('Use ' + kk + ' for G')
                g_params[kk] = data[kk]
    except IOError: 
        print('Use random initialization for G...')
        
    try: 
        data_d = np.load('./disent_result_d.npz') 
        print('Use saved initialization: disent_result_d.npz ...')
        for kk, pp in d_params.iteritems():
            if kk in data_d.keys() and d_params[kk].shape == data_d[kk].shape:
                print('Use ' + kk + ' for D')
                d_params[kk] = data_d[kk]
    except IOError: 
        print('Use random initialization for D...')    
   
    try: 
        data_cnn = np.load('./disent_result_cnn.npz') 
        print('Use pre-trained CNN for D...')
        for kk, pp in d_params.iteritems():
            if kk in data_cnn.keys() and d_params[kk].shape == data_cnn[kk].shape:
                print('Use ' + kk + 'for D')
                d_params[kk] = data_cnn[kk]
    except IOError: 
        print(' No pre-trained CNN...')  
        
        
    return d_params, g_params, s_params

def init_tparams(d_params, g_params, s_params, options):
    d_tparams = OrderedDict()
    g_tparams = OrderedDict()
    s_tparams = OrderedDict()    
    for kk, pp in d_params.iteritems():
        d_tparams[kk] = theano.shared(d_params[kk], name=kk)
    for kk, pp in g_params.iteritems():
        g_tparams[kk] = theano.shared(g_params[kk], name=kk)
    for kk, pp in s_params.iteritems():
        s_tparams[kk] = theano.shared(s_params[kk], name=kk)

    return d_tparams, g_tparams, s_tparams
   
""" Building model... """

def build_model(d_params, g_params, s_params, options):
    
    trng = RandomStreams(SEED)
    x = tensor.matrix('x', dtype='int32')   # n_sample * n_emb  where is n_word
    if options['debug']:
        x.tag.test_value = np.random.randint(2,size=(64,40)).astype('int32')   # batchsize * sent_len(n_word)  item: 0-voc_size
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.)) 
    
    # generative model part
    z = tensor.matrix('z', dtype='float32')  # n_batch * n_feature
    n_z = z.shape[0]
    
    
    n_samples = options['batch_size']
    n_words = options['n_words']
    n_x = d_params['Wemb'].shape[1] #embeding dim
    if options['shareLSTM']:
        h_decoder = decoder_g(g_params, z,options, max_step= options['max_step'] ,prefix='decoder_0')
    else:
        z_code = tensor.cast(z[:,0], dtype='int32')
        h_decoder = tensor.zeros([options['max_step'] , n_samples, options['n_h']])

        h_temp = []
        for idx in range(options['n_codes']):
            temp_idx = tensor.eq(z_code, idx).nonzero()[0]
            if options['sharedEmb']:
                h_decoder_temp = decoder_emb_from_d(g_params, d_params, z[:,1:] ,options, max_step= options['max_step'] ,prefix=_p('decoder',idx))
            else:
                h_decoder_temp = decoder_g(g_params, z[:,1:] ,options, max_step= options['max_step'] ,prefix=_p('decoder',idx))
            h_temp.append(h_decoder_temp)
            h_decoder = tensor.inc_subtensor(h_decoder[:,temp_idx,:],  h_temp[idx][:,temp_idx,:]) 
  
  
    
    #h_decoder = dropout(h_decoder, trng, use_noise)
    # reconstruct the original sentence
    shape_w = h_decoder.shape # n_step, n_sample , n_h
    h_decoder = h_decoder.reshape((shape_w[0]*shape_w[1], shape_w[2]))
    
    # pred_w: (n_steps * n_samples) * n_words 
    if options['sharedEmb']:
        Vhid = tensor.dot(g_params['Vhid'],d_params['Wemb'].T)
    else:
        Vhid = tensor.dot(g_params['Vhid'],g_params['Wemb'].T)   
    pred_w = tensor.dot(h_decoder, Vhid) + g_params['bhid']
    n_steps = shape_w[0]
    
    
    
    
    #  nondifferentiable
    if options['delta']> 1e-10:
        pred_w = tensor.switch(tensor.ge(pred_w,options['delta']), pred_w, 0)
    #pred_w = tensor.nnet.softmax(pred_w*options['L'])  
    max_w = tensor.max(pred_w, axis = 1, keepdims=True)
    e0 = tensor.exp((pred_w - max_w) * options['L']) 
    pred_w = e0 / tensor.sum(e0, axis = 1, keepdims=True)
    
    max_print = tensor.max(pred_w, axis = 1)
    max_print = max_print.reshape((n_steps, n_samples)).dimshuffle(1,0)
 
    pred_w = pred_w.reshape((n_steps, n_samples,n_words )).dimshuffle(1, 0, 2)  # reshape need parenthesis
    

    if options['force_cut'] == 'cut':
        rng_temp = tensor.minimum(-tensor.sum(tensor.log(trng.uniform((n_samples,6))),axis = 1)*3.3,options['max_step']-5)
        rng_length = tensor.floor(rng_temp).astype('int32')   #gamma(6,3.3)
        # pred_mask = tensor.zeros(pred_w.shape)
        period = options['period']
        # should use set values
        for i in xrange(n_samples):
            pred_w = tensor.set_subtensor(pred_w[i,rng_length[i]:,:] , 0)
            pred_w = tensor.set_subtensor(pred_w[i,rng_length[i],period] , 1)
            pred_w = tensor.set_subtensor(pred_w[i,(rng_length[i]+1):,0] , 1)
    elif options['force_cut'] == 'strip':
        for i in xrange(n_samples):
            pred_w = tensor.set_subtensor(pred_w[i,options['max_step']-1,0] , 1)
            idx_end = theano.tensor.eq(tensor.argmax(pred_w[i,:,:],axis=1) , 0).nonzero()[0][0]
            pred_w = tensor.set_subtensor(pred_w[i,(idx_end+1):,0] , 1)
            pred_w = tensor.set_subtensor(pred_w[i,(idx_end+1):,1:] , 0)
    
    
    
    pad = max(options['filter_hs']) - 1
    end_mat = tensor.concatenate([tensor.ones([n_samples,pad,1]), tensor.zeros([n_samples,pad,n_words-1])],axis=2)
    pred_w = tensor.concatenate([end_mat, pred_w, end_mat],axis = 1)
    
    
    
    n_steps = n_steps+2*pad
    pred_w = pred_w.reshape((n_steps*n_samples,n_words))
    
    

    # should be d's embeding
    fake_input = tensor.dot(pred_w, d_params['Wemb'])
    
    

    #  real[ 64   1  68 300] fake[ 64   1  41 300]   
    fake_input = fake_input.reshape((n_samples,1,n_steps,d_params['Wemb'].shape[1]))    #(64,1,  )
    use_noise2 = theano.shared(numpy_floatX(0.)) 
    fake_input = dropout(fake_input, trng, use_noise2)

    
    # fake feature output
    fake_outputs1 = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(d_params, fake_input,filter_shape, pool_size, options, prefix=_p('cnn_d',i))                          
        fake_output1 = conv_layer
        fake_outputs1.append(fake_output1)

    fake_output1 = tensor.concatenate(fake_outputs1,1) # should be 64*900
    if options['batch_norm']:
        fake_output1 = batch_norm(d_params, fake_output1, options, prefix='fake')             


    if options['cnn_activation'] == 'tanh':
        fake_pred = mlp_layer_linear(d_params, fake_output1, prefix='dis_d')
    elif options['cnn_activation'] == 'linear':
        fake_pred = mlp_layer_linear(d_params, tensor.tanh(fake_output1), prefix='dis_d')#
        
    if not options['wgan']:
        fake_pred = tensor.nnet.sigmoid(fake_pred) * (1 - 2*options['label_smoothing']) + options['label_smoothing']  
    
    
    # for reverse model 
    # if options['reverse']:
    fake_recon = mlp_layer_tanh(d_params, fake_output1, prefix='recon')
    r_t  = fake_recon/2.0 + .5
    z_t = z/2.0 + .5
    r_cost=(-z_t*tensor.log(r_t+0.0001)-(1.-z_t)*tensor.log(1.0001-r_t)).sum()/n_samples/n_z
    
    
    
    
    # Proposal nets (for infogan)
    fake_outputs2 = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(g_params, fake_input,filter_shape, pool_size, options, prefix=_p('cnn_d',i))                          
        fake_output2 = conv_layer
        fake_outputs2.append(fake_output2)
    fake_output2 = tensor.concatenate(fake_outputs2,1) # should be 64*900     # why it is 64*0???

    # check whether to use softmax or tanh
    fake_propose = mlp_layer_tanh(g_params, fake_output2, prefix='dis_q')
    fake_propose = (fake_propose + 1)/2
    fake_propose = tensor.log(fake_propose) 
    z_code = tensor.cast(z[:,0], dtype='int32') 
    z_index = tensor.arange(n_z)
    fake_logent = fake_propose[z_index ,z_code]
    l_I = tensor.sum(fake_logent) 
    

    # Wemb: voc_size(n_words) * n_emb       64* 1* 40 *48
    real_input = d_params['Wemb'][tensor.cast(x.flatten(),dtype='int32')].reshape((x.shape[0],1,x.shape[1],d_params['Wemb'].shape[1])) # n_sample,1,n_length,n_emb
    real_input = dropout(real_input, trng, use_noise2)
 
    real_outputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer2 = encoder(d_params, real_input,filter_shape, pool_size, options, prefix=_p('cnn_d',i))                          
        real_output = conv_layer2
        real_outputs.append(real_output)
    real_output = tensor.concatenate(real_outputs,1)

    if options['batch_norm']:
       real_output=batch_norm(d_params, real_output, options, prefix='real')  
       
     
     
    if options['cnn_activation'] == 'tanh':
        real_pred = mlp_layer_linear(d_params, real_output, prefix='dis_d')  
    elif options['cnn_activation'] == 'linear':
        real_pred = mlp_layer_linear(d_params, tensor.tanh(real_output), prefix='dis_d')
           
    if not options['wgan']:
        real_pred = tensor.nnet.sigmoid(real_pred) * (1 - 2*options['label_smoothing']) + options['label_smoothing']  

    
    #Compute for KDE    
    mu=real_output
    X=fake_output1
    KDE=cal_nkde(X,mu,options['kde_sigma'])

    #calculate KDE on real_input and fake_input
    X_i = fake_input.reshape((n_samples,n_steps*d_params['Wemb'].shape[1]))
    mu_i = real_input.reshape((n_samples,n_steps*d_params['Wemb'].shape[1]))
    KDE_input=cal_nkde(X_i,mu_i,options['kde_sigma'])

    # sufficient statistics
    cur_size = s_params['seen_size']*1.0
    identity = tensor.eye(options['n_z'])*options['diag']
    fake_mean = tensor.mean(fake_output1,axis=0)
    real_mean = tensor.mean(real_output,axis=0)
    fake_xx = tensor.dot(fake_output1.T, fake_output1)
    real_xx = tensor.dot(real_output.T, real_output)
    acc_fake_xx = (s_params['acc_fake_xx']*cur_size + fake_xx)/(cur_size + n_samples) 
    acc_real_xx = (s_params['acc_real_xx']*cur_size + real_xx)/(cur_size + n_samples) 
    acc_fake_mean = (s_params['acc_fake_mean']*cur_size + fake_mean*n_samples)/(cur_size + n_samples)
    acc_real_mean = (s_params['acc_real_mean']*cur_size + real_mean*n_samples)/(cur_size + n_samples)
    
    cov_fake = acc_fake_xx - tensor.dot(acc_fake_mean.dimshuffle(0, 'x'), acc_fake_mean.dimshuffle(0, 'x').T)  +identity
    cov_real = acc_real_xx - tensor.dot(acc_real_mean.dimshuffle(0, 'x'), acc_real_mean.dimshuffle(0, 'x').T)  +identity
      
    cov_fake_inv = tensor.nlinalg.matrix_inverse(cov_fake)
    cov_real_inv = tensor.nlinalg.matrix_inverse(cov_real)


    if options['feature_match'] == 'moment':
        temp1 = ((fake_mean - real_mean)**2).sum()
        fake_obj = temp1 
        
    elif options['feature_match'] == 'JSD_acc':


        temp1 = tensor.nlinalg.trace(tensor.dot(cov_fake_inv,cov_real) + tensor.dot(cov_real_inv,cov_fake))
        temp2 = tensor.dot( tensor.dot((acc_fake_mean - acc_real_mean) , (cov_fake_inv + cov_real_inv)), (acc_fake_mean - acc_real_mean).T)        
               
        fake_obj = temp1 + temp2 
        
    elif options['feature_match'] == 'mmd':
        #### too many nodes, use scan ####
        kxx, kxy, kyy = 0, 0, 0
        dividend = 1
        dist_x, dist_y = fake_output1/dividend, real_output/dividend
        x_sq = tensor.sum(dist_x**2,axis=1).dimshuffle(0, 'x')    #  64*1
        y_sq = tensor.sum(dist_y**2,axis=1).dimshuffle(0, 'x')    #  64*1
        tempxx = -2*tensor.dot(dist_x,dist_x.T) + x_sq + x_sq.T  # (xi -xj)**2
        tempxy = -2*tensor.dot(dist_x,dist_y.T) + x_sq + y_sq.T  # (xi -yj)**2     
        tempyy = -2*tensor.dot(dist_y,dist_y.T) + y_sq + y_sq.T  # (yi -yj)**2
           
        
        for sigma in options['sigma_range']:
            kxx += tensor.mean(tensor.exp(-tempxx/2/(sigma**2)))
            kxy += tensor.mean(tensor.exp(-tempxy/2/(sigma**2)))
            kyy += tensor.mean(tensor.exp(-tempyy/2/(sigma**2)))
        
        fake_obj = tensor.sqrt(kxx + kyy - 2*kxy)
        
    elif options['feature_match'] == 'mmd_cov':
        kxx, kxy, kyy = 0, 0, 0
        cov_sum = (cov_fake + cov_real)/2
        cov_sum_inv = tensor.nlinalg.matrix_inverse(cov_sum)
        
        dividend = 1
        dist_x, dist_y = fake_output1/dividend, real_output/dividend
        cov_inv_mat = cov_sum_inv
        x_sq = tensor.sum(tensor.dot(dist_x, cov_inv_mat) * dist_x, axis=1).dimshuffle(0, 'x')
        y_sq = tensor.sum(tensor.dot(dist_y, cov_inv_mat) * dist_y, axis=1).dimshuffle(0, 'x')
        
        tempxx = -2*tensor.dot(tensor.dot(dist_x, cov_inv_mat), dist_x.T) + x_sq + x_sq.T  # (xi -xj)**2
        tempxy = -2*tensor.dot(tensor.dot(dist_x, cov_inv_mat), dist_y.T) + x_sq + y_sq.T  # (xi -yj)**2     
        tempyy = -2*tensor.dot(tensor.dot(dist_y, cov_inv_mat), dist_y.T) + y_sq + y_sq.T  # (yi -yj)**2
           
        
        for sigma in options['sigma_range']:
            kxx += tensor.mean(tensor.exp(-tempxx/2/(sigma**2)))
            kxy += tensor.mean(tensor.exp(-tempxy/2/(sigma**2)))
            kyy += tensor.mean(tensor.exp(-tempyy/2/(sigma**2)))
        fake_obj = tensor.sqrt(kxx + kyy - 2*kxy)    
        
        
    elif options['feature_match'] == 'mmd_ld':
        
        kxx, kxy, kyy = 0, 0, 0
        real_mmd = mlp_layer_tanh(d_params, real_output, prefix='dis_mmd') 
        fake_mmd = mlp_layer_tanh(d_params, fake_output1, prefix='dis_mmd') 

        dividend = options['dim_mmd']  # for numerical stability & scale with 
        dist_x, dist_y = fake_mmd/dividend, real_mmd/dividend
        x_sq = tensor.sum(dist_x**2,axis=1).dimshuffle(0, 'x')    #  64*1
        y_sq = tensor.sum(dist_y**2,axis=1).dimshuffle(0, 'x')    #  64*1
        tempxx = -2*tensor.dot(dist_x,dist_x.T) + x_sq + x_sq.T  # (xi -xj)**2
        tempxy = -2*tensor.dot(dist_x,dist_y.T) + x_sq + y_sq.T  # (xi -yj)**2     
        tempyy = -2*tensor.dot(dist_y,dist_y.T) + y_sq + y_sq.T  # (yi -yj)**2
           
        
        for sigma in options['sigma_range']:
            kxx += tensor.exp(-tempxx/2/sigma).sum()
            kxy += tensor.exp(-tempxy/2/sigma).sum()
            kyy += tensor.exp(-tempyy/2/sigma).sum()

        fake_obj = tensor.sqrt(kxx + kyy - 2*kxy)
        
    elif options['feature_match'] == 'mmd_h':
        #### too many nodes, use scan ####
        
        kxx, kxy, kyy = 0, 0, 0
        
        if options['cnn_activation'] == 'tanh':
            fake_mmd = middle_layer(d_params, fake_output1, prefix='dis_d')
        elif options['cnn_activation'] == 'linear':
            fake_mmd = middle_layer(d_params, tensor.tanh(fake_output1), prefix='dis_d')#
            
        if options['cnn_activation'] == 'tanh':
            real_mmd = middle_layer(d_params, real_output, prefix='dis_d')
        elif options['cnn_activation'] == 'linear':
            real_mmd = middle_layer(d_params, tensor.tanh(real_output), prefix='dis_d')#    

        dividend = 1
        dist_x, dist_y = fake_mmd/dividend, real_mmd/dividend
        x_sq = tensor.sum(dist_x**2,axis=1).dimshuffle(0, 'x')    #  64*1
        y_sq = tensor.sum(dist_y**2,axis=1).dimshuffle(0, 'x')    #  64*1
        tempxx = -2*tensor.dot(dist_x,dist_x.T) + x_sq + x_sq.T  # (xi -xj)**2
        tempxy = -2*tensor.dot(dist_x,dist_y.T) + x_sq + y_sq.T  # (xi -yj)**2     
        tempyy = -2*tensor.dot(dist_y,dist_y.T) + y_sq + y_sq.T  # (yi -yj)**2
           
        
        for sigma in options['sigma_range']:
            kxx += tensor.mean(tensor.exp(-tempxx/2/(sigma**2)))
            kxy += tensor.mean(tensor.exp(-tempxy/2/(sigma**2)))
            kyy += tensor.mean(tensor.exp(-tempyy/2/(sigma**2)))
        fake_obj = tensor.sqrt(kxx + kyy - 2*kxy)
        

    else:
        fake_obj = - tensor.log(fake_pred + 1e-6).sum() / n_z   

    if options['wgan']:
        gan_cost_d =  fake_pred.sum() / n_z - real_pred.sum() / n_samples 
        gan_cost_g = -fake_pred.sum() / n_z + 0*((fake_mean - acc_real_mean)**2).sum()
    else:
        gan_cost_d = - tensor.log(1-fake_pred + 1e-6).sum() / n_z - tensor.log(real_pred + 1e-6).sum() / n_samples 
        gan_cost_g = fake_obj 
        
    #result4 = fake_obj  
    d_cost =  gan_cost_d - options['lambda_fm']*fake_obj + options['lambda_recon']*r_cost + options['lambda_q'] * l_I /n_z
    g_cost = gan_cost_g - options['lambda_q'] * l_I / n_z
    
    #result1, result2, result4, result5, result6 = x_sq, y_sq, tempxx, tempxy, tempyy    
    
    result1 = tensor.mean(real_pred) # goes to nan
    result2 = tensor.mean(fake_pred) # goes to nan
    result3 = tensor.argmax(pred_w,axis=1).reshape([n_samples, n_steps])
    result4 = tensor.nlinalg.trace(tensor.dot(cov_fake_inv,cov_real) + tensor.dot(cov_real_inv,cov_fake))
    result5 = max_print[0] #mu  #tensor.dot( tensor.dot((acc_fake_mean - acc_real_mean) , (cov_fake_inv + cov_real_inv)), (acc_fake_mean - acc_real_mean).T)
    result6 = ((fake_mean - real_mean)**2).sum()
    


              
    return use_noise, use_noise2, x ,z, d_cost, g_cost, r_cost, fake_recon, acc_fake_xx, acc_real_xx, acc_fake_mean, acc_real_mean, result1, result2, result3, result4, result5 ,result6, KDE, KDE_input
