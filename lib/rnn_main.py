#!/usr/bin/env python

import sys
sys.path.append("/data/lisatmp3/kenan/dreamprop/lib")

import cPickle as pickle
import gzip
from loss import accuracy, crossent, expand, nll
import theano
import theano.tensor as T
import numpy.random as rng
import lasagne
import numpy as np
from random import randint
import time

import nn_layers

from nn_layers import param_init_lnlstm, lnlstm_layer, param_init_lngru, lngru_layer, param_init_fflayer, fflayer

from utils import init_tparams

from ptb_data import get_batch

sys.setrecursionlimit(9999999)

#dataset = "ptb"
dataset = "seqmnist"
#dataset = "vocal"

print "dataset", dataset

if dataset == "ptb":
    from ptb_data import get_batch

    n_feat = 50
    n_target = 50

elif dataset == "seqmnist":
    from seqmnist_data import get_batch

    num_steps = 783

    n_feat = 2
    n_target = 2

srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))


print "doing deep bp"
print "using 1 layer forward net"
print "Number of steps", num_steps


def init_params_forward():

    p = {}

    param_init_lngru({}, params=p, prefix='gru1', nin=512, dim=512)

    #param_init_lngru({}, params=p, prefix='gru2', nin=512, dim=512)

    tparams = nn_layers.init_tparams(p)

    tparams["w1"] = theano.shared(0.01 * rng.normal(0,1,size=(2,512)).astype('float32'))

    tparams["w2"] = theano.shared(0.01 * rng.normal(0,1,size=(512+2,128)).astype('float32'))
    tparams["b2"] = theano.shared(0.0 * rng.normal(0,1,size=(128,)).astype('float32'))

    tparams["Wy"] = theano.shared(0.01 * rng.normal(0,1,size=(128,2)).astype('float32'))
    tparams["by"] = theano.shared(0.0 * rng.normal(0,1,size=(2,)).astype('float32'))

    return tparams

def init_params_synthmem():

    p = {}

    param_init_fflayer({},params=p,prefix="synthmem_fc1", nin=512, nout=512)

    tparams = nn_layers.init_tparams(p)

    return tparams

def join2(a,b):
        return T.concatenate([a,b], axis = 1)

def join3(a,b,c):
        return T.concatenate([a,b,c], axis = 1)


def ln(inp):
    return (inp - T.mean(inp,axis=1,keepdims=True)) / (0.001 + T.std(inp,axis=1,keepdims=True))

def forward(p, h, x_true, y_true):


    print "USING LAYER NORM"

    emb = T.dot(x_true, p['w1'])

    h_next1 = lngru_layer(p,emb,{},prefix='gru1',mask=None,one_step=True,init_state=h[:,:512],backwards=False)

    #h_next2 = lngru_layer(p,h_next1[0],{},prefix='gru2',mask=None,one_step=True,init_state=h[:,512:],backwards=False)

    hout = join2(h_next1[0], x_true)

    h2 = T.tanh(ln(T.dot(hout, p['w2']) + p['b2']))

    y_est = T.nnet.softmax(T.dot(h2, p['Wy']) + p['by'])

    loss = nll(y_est, y_true,2)

    acc = accuracy(y_est, y_true)

    return h_next1[0], y_est, loss, acc

params_forward = init_params_forward()
params_synthmem = init_params_synthmem()

def synthmem(p, h_next, h_last):

    h_out = fflayer(p,h_next,{},prefix='synthmem_fc1',activ='lambda x: tensor.tanh(x)')

    rec_loss_h = T.mean(T.abs_(h_out - h_last))

    return h_out, rec_loss_h

'''
Set up the forward method and the synthmem_method
'''


'''
BPTT method: 
    -Run forward pass for many steps.  
    -Optimize total loss wrt all params.  
'''

x = T.imatrix()
y = T.imatrix()
h_initial = T.matrix()

def one_step(xval,yval,total_loss,total_acc,hval,yel,h_rec_loss_last):
    h_next,y_est,loss,acc = forward(params_forward, hval, expand(xval,2).astype('float32'),yval)

    h_last_rec, h_rec_loss = synthmem(params_synthmem, h_next, hval)

    return [(loss).astype('float32'),(acc).astype('float32'),h_next.astype('float32'),y_est.argmax(axis=1).astype('float32'),h_rec_loss]

h_initial_shared = theano.shared(0.0 * rng.normal(size = (1,512)).astype('float32'))
h_next = h_initial + T.addbroadcast(h_initial_shared, 0)

y_est = theano.shared(np.zeros(shape=(64,)).astype('float32'))

h_rec_loss = theano.shared(np.array(0.0).astype('float32'))

scan_res, scan_updates = theano.scan(fn=one_step, sequences=[x.T,y.T], outputs_info=[np.asarray(0.0,dtype='float32'),np.asarray(0.0,dtype='float32'),h_next,y_est, h_rec_loss])

total_loss, total_acc, h_final, yest, h_rec_loss = scan_res

print "mult rec loss 100"
loss_use = total_loss.mean() + 100.0 * h_rec_loss.mean()

lr = 0.0001
print "learning rate", lr
rule = lasagne.updates.adam
print "rule", rule
updates = rule(loss_use, params_forward.values() + [h_initial_shared] + params_synthmem.values(), learning_rate=lr)

t0 = time.time()

train_bptt = theano.function(inputs = [x,y,h_initial], outputs = [h_final[-1], total_loss.sum()/64.0, total_acc.mean(),yest], updates = updates)
print "time to compile", time.time() - t0

evaluate = theano.function(inputs = [x,y,h_initial], outputs = [h_final[-1], total_loss.sum()/64.0, total_acc.mean(),yest,h_rec_loss.sum()/64.0])

ta = []
tc = []
hlst = []

for iteration in xrange(0,100000):

    x,y = get_batch("train")

    t0 = time.time()

    h_initial = 0.0 * np.random.normal(size=(64,512)).astype('float32')

    lossl = []
    accl = []

    t0 = time.time()
    for step in range(0,1):
        h_initial,loss,acc,_ = train_bptt(x[:,step*783:(step+1)*783],y[:,step*783:(step+1)*783],h_initial)
        lossl.append(loss)
        accl.append(acc)

    #print "yest shape", yest.shape

    #print loss,acc

    if iteration % 100 == 0:
        print "iteration", iteration
        #print time.time() - t0, "TIME TO TRAIN ONE EXAMPLE"

        print "loss train", sum(lossl)
        print "acc train", sum(accl)/len(accl)

        #print "true", y[0,:].tolist()
        #print "est", yest[:,0].tolist()

        lossl = []
        accl = []
        hreclossl = []
        for j in range(0,50):
            x,y = get_batch("test",64)

            h_initial = 0.0 * np.random.normal(size=(64,512)).astype('float32')

            #t0 = time.time()
            h_out,loss,acc,yest,h_rec_loss = evaluate(x,y,h_initial)
            #print time.time() - t0, "time to do validation update"

            lossl.append(loss)
            accl.append(acc)
            hreclossl.append(h_rec_loss)

        print "TEST LOSS", iteration, sum(lossl)/len(lossl)
        print "TEST ACC", iteration, sum(accl)/len(accl)
        print "TEST RECLOSS", iteration, sum(hreclossl)/len(hreclossl)




