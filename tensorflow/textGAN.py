# -*- coding: utf-8 -*-
"""
Yizhe Zhang

TextGAN
"""
## 152.3.214.203/6006

import os
GPUID = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
#from tensorflow.contrib import metrics
#from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
#from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import cPickle
import numpy as np
import os
import scipy.io as sio
from math import floor
import pdb

from model import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save, \
    prepare_for_bleu, cal_BLEU, sent2idx, _clip_gradients_seperate_norm
from denoise import *

profile = False
#import tempfile
#from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
#tf.logging.verbosity(1)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')





class Options(object):
    def __init__(self):
        self.dis_steps = 10
        self.gen_steps = 1
        self.fix_emb = False
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = False  # reuse cnn for discrimination
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_rnn' #'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 41
        self.n_words = 4391
        self.filter_shape = 5
        self.filter_size = 300
        self.multiplier = 2
        self.embed_size = 300
        self.latent_size = 128
        self.lr = 1e-5

        self.layer = 3
        self.stride = [2, 2, 2]   # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 256
        self.max_epochs = 100
        self.n_gan = 128  # self.filter_size * 3
        self.L = 1000
        
        self.rnn_share_emb = True
        self.additive_noise_lambda = 0.0
        self.bp_truncation = None
        self.n_hid = 100

        self.optimizer = 'Adam' #tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None #None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 0.99
        self.relu_w = False

        self.save_path = "./save/" + "bird_" + str(self.n_gan) + "_dim_" + self.model + "_" + self.substitution + str(self.permutation) 
        self.log_path = "./log"
        self.print_freq = 10
        self.valid_freq = 100
        self.sigma_range = [2]

        # batch norm & dropout
        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 1

        self.discrimination = False
        self.H_dis = 300

        self.sent_len = self.maxlen + 2*(self.filter_shape-1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape)/self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape)/self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)
        self.sentence = self.maxlen - 1
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

def discriminator(x, opt, prefix = 'd_', is_prob = False, is_reuse = None):
    W_norm_d = embedding_only(opt, prefix = prefix, is_reuse = is_reuse)   # V E
    H = encoder(x, W_norm_d, opt, prefix = prefix + 'enc_',  is_prob=is_prob, is_reuse = is_reuse)
    logits = discriminator_2layer(H, opt, is_reuse = is_reuse)

    return logits, H

def encoder(x, W_norm_d, opt, prefix = 'd_', is_prob = False, is_reuse = None, is_padded = True):
    if is_prob:
        x_emb = tf.tensordot(x, W_norm_d, [[2],[0]])
    else:
        x_emb = tf.nn.embedding_lookup(W_norm_d, x)   # batch L emb
    if not is_padded:  # pad the input with pad_emb
        pad_emb = tf.expand_dims(tf.expand_dims(W_norm_d[0],0),0) # 1*v
        x_emb = tf.concat([tf.tile(pad_emb, [opt.batch_size, opt.filter_shape-1, 1]), x_emb],1)

    x_emb = tf.expand_dims(x_emb,3)   # batch L emb 1
    #bp()
    if opt.layer == 3:
        H = conv_model_3layer(x_emb, opt, prefix = prefix, is_reuse = is_reuse)
    else:
        H = conv_model(x_emb, opt, prefix = prefix, is_reuse = is_reuse)
    return tf.squeeze(H)



def textGAN(x, opt):
    
    #res = {}
    res_ = {}
        
    with tf.variable_scope("pretrain"): 
        #z = tf.random_uniform([opt.batch_size, opt.latent_size], minval=-1.,maxval=1.)
        z = tf.random_normal([opt.batch_size, opt.latent_size])
        W_norm = embedding_only(opt, is_reuse = None) 
        _, syn_sent, logits = lstm_decoder_embedding(z, tf.ones_like(x), W_norm, opt, add_go = True, feed_previous=True, is_reuse=None, is_softargmax = True, is_sampling = False)
        prob = [tf.nn.softmax(l*opt.L) for l in logits]
        prob = tf.stack(prob,1)

        # _, syn_onehot, rec_sent, _ = lstm_decoder_embedding(z, x_org, W_norm, opt)
        # x_emb_fake = tf.tensordot(syn_onehot, W_norm, [[2],[0]])
        # x_emb_fake = tf.expand_dims(x_emb_fake, 3)

    with tf.variable_scope("d_net"): 
        logits_real, H_real = discriminator(x, opt)

        


        ## Real Trail
        # x_emb, W_norm = embedding(x, opt, is_reuse = None)  # batch L emb
        # x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1
        # H_enc, res = conv_encoder(x_emb, opt, res, is_reuse = None)
        
    with tf.variable_scope("d_net"):
        logits_fake, H_fake = discriminator(prob, opt, is_prob = True, is_reuse = True)


        # H_enc_fake, res_ = conv_encoder(x_emb_fake, is_train, opt, res_, is_reuse=True)
        # logits_real = discriminator_2layer(H_enc, opt)
        # logits_syn = discriminator_2layer(H_enc_fake, opt, is_reuse=True)
    
    res_['syn_sent'] = syn_sent
    res_['real_f'] = tf.squeeze(H_real)
    # Loss
    
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits = logits_real)) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_fake), logits = logits_fake))
  
    fake_mean = tf.reduce_mean(H_fake,axis=0)
    real_mean = tf.reduce_mean(H_real,axis=0)
    mean_dist = tf.sqrt(tf.reduce_mean((fake_mean - real_mean)**2))
    res_['mean_dist'] = mean_dist

    # cov_fake = acc_fake_xx - tensor.dot(acc_fake_mean.dimshuffle(0, 'x'), acc_fake_mean.dimshuffle(0, 'x').T)  +identity
    # cov_real = acc_real_xx - tensor.dot(acc_real_mean.dimshuffle(0, 'x'), acc_real_mean.dimshuffle(0, 'x').T)  +identity
      
    # cov_fake_inv = tensor.nlinalg.matrix_inverse(cov_fake)
    # cov_real_inv = tensor.nlinalg.matrix_inverse(cov_real)
    #tensor.nlinalg.trace(tensor.dot(cov_fake_inv,cov_real) + tensor.dot(cov_real_inv,cov_fake))
        
    GAN_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))
    MMD_loss = compute_MMD_loss(tf.squeeze(H_fake), tf.squeeze(H_real), opt)
    G_loss = mean_dist #MMD_loss # + tf.reduce_mean(GAN_loss) # mean_dist #
    res_['mmd'] = MMD_loss
    res_['gan'] = tf.reduce_mean(GAN_loss)
    # *tf.cast(tf.not_equal(x_temp,0), tf.float32)
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)
    summaries = [
                "learning_rate",
                "G_loss",
                "D_loss"
                # "gradients",
                # "gradient_norm",
                ]
    global_step = tf.Variable(0, trainable=False)
    
    all_vars = tf.trainable_variables()
    g_vars = [var for var in all_vars if 
                  var.name.startswith('pretrain')]
    d_vars = [var for var in all_vars if
              var.name.startswith('d_')]
    print [g.name for g in g_vars]
    generator_op = layers.optimize_loss(
        G_loss,
        global_step = global_step,
        #aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        #framework.get_global_step(),
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        learning_rate_decay_fn=lambda lr,g: tf.train.exponential_decay(learning_rate=lr, global_step = g, decay_rate=opt.decay_rate, decay_steps=3000),
        learning_rate=opt.lr,
        variables = g_vars,
        summaries = summaries
        )
    
    discriminator_op = layers.optimize_loss(
        D_loss,
        global_step = global_step,
        #aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        #framework.get_global_step(),
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        learning_rate_decay_fn=lambda lr,g: tf.train.exponential_decay(learning_rate=lr, global_step = g, decay_rate=opt.decay_rate, decay_steps=3000),
        learning_rate=opt.lr,
        variables = d_vars,
        summaries = summaries
        )

    # optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr)  # Or another optimization algorithm.
    # train_op = optimizer.minimize(
    #     loss,
    #     aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)


    return res_, G_loss, D_loss, generator_op, discriminator_op


def run_model(opt, train, val, ixtoword):


    try:
        params = np.load('./param_g.npz')
        if params['Wemb'].shape == (opt.n_words, opt.embed_size):
            print('Use saved embedding.')
            opt.W_emb = params['Wemb']
        else:
            print('Emb Dimension mismatch: param_g.npz:'+ str(params['Wemb'].shape) + ' opt: ' + str((opt.n_words, opt.embed_size)))
            opt.fix_emb = False
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        x_org_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        is_train_ = tf.placeholder(tf.bool, name='is_train_')
        res_, g_loss_, d_loss_, gen_op, dis_op = textGAN(x_, opt)
        merged = tf.summary.merge_all()
        # opt.is_train = False
        # res_val_, loss_val_, _ = auto_encoder(x_, x_org_, opt)
        # merged_val = tf.summary.merge_all()

    #tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006
    #writer = tf.train.SummaryWriter(opt.log_path, graph=tf.get_default_graph())


    uidx = 0
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=tf.GraphOptions(build_cost_model=1))
    #config = tf.ConfigProto(device_count={'GPU':0})
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()



    run_metadata = tf.RunMetadata()


    with tf.Session(config = config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                #pdb.set_trace()

                t_vars = tf.trainable_variables()
                #print([var.name[:-2] for var in t_vars])
                loader = restore_from_save(t_vars, sess, opt)


            except Exception as e:
                print(e)
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        for epoch in range(opt.max_epochs):
            print("Starting epoch %d" % epoch)
            # if epoch >= 10:
            #     print("Relax embedding ")
            #     opt.fix_emb = False
            #     opt.batch_size = 2
            kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
                sents = [train[t] for t in train_index]

                sents_permutated = add_noise(sents, opt)

                #sents[0] = np.random.permutation(sents[0])
                x_batch = prepare_data_for_cnn(sents_permutated, opt) # Batch L
                d_loss = 0
                g_loss = 0
                if profile:
                    if uidx % opt.dis_steps == 0:
                        _, d_loss = sess.run([dis_op, d_loss_], feed_dict={x_: x_batch},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
                    if uidx % opt.gen_steps == 0:
                        _, g_loss = sess.run([gen_op, g_loss_], feed_dict={x_: x_batch},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
                else:
                    if uidx % opt.dis_steps == 0:
                        _, d_loss = sess.run([dis_op, d_loss_], feed_dict={x_: x_batch})
                    if uidx % opt.gen_steps == 0:
                        _, g_loss = sess.run([gen_op, g_loss_], feed_dict={x_: x_batch})
                
                
                


                if uidx % opt.valid_freq == 0:
                    is_train = True
                    valid_index = np.random.choice(len(val), opt.batch_size)
                    val_sents = [val[t] for t in valid_index]

                    val_sents_permutated = add_noise(val_sents, opt)

                    x_val_batch = prepare_data_for_cnn(val_sents_permutated, opt)

                    d_loss_val = sess.run(d_loss_, feed_dict={x_: x_val_batch})
                    g_loss_val = sess.run(g_loss_, feed_dict={x_: x_val_batch})


                    res = sess.run(res_, feed_dict={x_: x_val_batch})
                    print("Validation d_loss %f, g_loss %f  mean_dist %f" % (d_loss_val, g_loss_val, res['mean_dist']))
                    print "Sent:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()
                    print("MMD loss %f, GAN loss %f" % (res['mmd'], res['gan']))
                    np.savetxt('./text/rec_val_words.txt', res['syn_sent'], fmt='%i', delimiter=' ')
                    if opt.discrimination:
                        print ("Real Prob %f Fake Prob %f" % (res['prob_r'], res['prob_f']))
                    

                    val_set = [prepare_for_bleu(s) for s in val_sents]
                    [bleu2s, bleu3s, bleu4s] = cal_BLEU([prepare_for_bleu(s) for s in res['syn_sent']], {0: val_set})
                    print 'Val BLEU (2,3,4): ' + ' '.join([str(round(it, 3)) for it in (bleu2s, bleu3s, bleu4s)])

                    summary = sess.run(merged, feed_dict={x_: x_val_batch})
                    test_writer.add_summary(summary, uidx)
                    



                
                if uidx % opt.print_freq == 0:
                    #pdb.set_trace()
                    res = sess.run(res_, feed_dict={x_: x_batch})
                    median_dis = np.sqrt(np.median([((x-y)**2).sum() for x in res['real_f'] for y in res['real_f']]))
                    print("Iteration %d: d_loss %f, g_loss %f, mean_dist %f, realdist median %f" % (uidx, d_loss, g_loss, res['mean_dist'], median_dis))                    
                    np.savetxt('./text/rec_train_words.txt', res['syn_sent'], fmt='%i', delimiter=' ')
                    print "Sent:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()

                    summary = sess.run(merged, feed_dict={x_: x_batch})
                    train_writer.add_summary(summary, uidx)
                    # print res['x_rec'][0][0]
                    # print res['x_emb'][0][0]
                    if profile:
                        tf.contrib.tfprof.model_analyzer.print_model_analysis(
                        tf.get_default_graph(),
                        run_meta=run_metadata,
                        tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

            saver.save(sess, opt.save_path, global_step=epoch)



def main():
    #global n_words
    # Prepare training and testing data
    #loadpath = "./data/three_corpus_small.p"
    loadpath = "./real_cotra.txt"
    x = np.loadtxt(loadpath)
    train, val = x[:88550], x[88550:]
    ixtoword, _ = cPickle.load(open('vocab_cotra.pkl','rb'))
    ixtoword = {i:x for i,x in enumerate(ixtoword)}
    opt = Options()

    print dict(opt)
    print('Total words: %d' % opt.n_words)
    
    run_model(opt, train, val, ixtoword)

    # model_fn = auto_encoder
    # ae = learn.Estimator(model_fn=model_fn)
    # ae.fit(train, opt , steps=opt.max_epochs)


#
# def main(argv=None):
#     learn_runner.run(experiment_fn, FLAGS.train_dir)

if __name__ == '__main__':
    main()
