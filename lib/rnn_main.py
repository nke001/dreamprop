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

    p['W1'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024+n_feat,1024)).astype('float32'))
    #p['W2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['Wy'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,n_target)).astype('float32'))
    #p['Wo'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024,1024)).astype('float32'))


    return p

def init_params_synthmem():

    p = {}

    p['Wh'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,2048)).astype('float32'))
    p['Wh2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,2048,1024)).astype('float32'))

    p['Wx'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,2048)).astype('float32'))
    p['Wx2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,2048,n_feat)).astype('float32'))

    p['Wy1'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,1024)).astype('float32'))
    p['Wy2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,n_target)).astype('float32'))

    p['bh'] = theano.shared(0.03 * rng.normal(0,1,size=(1,2048,)).astype('float32'))
    p['bh2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,)).astype('float32'))
    
    p['bx'] = theano.shared(0.03 * rng.normal(0,1,size=(1,2048,)).astype('float32'))
    p['bx2'] = theano.shared(0.03 * rng.normal(0,1,size=(1,n_feat,)).astype('float32'))

    p['by1'] = theano.shared(0.03 * rng.normal(0,1,size=(1,1024,)).astype('float32'))

    return p

def join2(a,b):
        return T.concatenate([a,b], axis = 1)

def ln(inp):
    return (inp - T.mean(inp,axis=1,keepdims=True)) / (0.001 + T.std(inp,axis=1,keepdims=True))


def forward(p, h, x_true, y_true, i):

    i *= 0

    inp = join2(h, x_true)

    h1 = T.nnet.relu(ln(T.dot(inp, p['W1'][i])), alpha=0.02)
    #h2 = T.nnet.relu(ln(T.dot(h1, p['W2'])), alpha=0.02)
    h2 = h1

    y_est = T.nnet.softmax(T.dot(h2, p['Wy'][i]))

    #h_next = T.dot(h2, p['Wo'][i])
    h_next = h1

    loss = nll(y_est, y_true,2)

    acc = accuracy(y_est, y_true)

    return h_next, y_est, loss, acc

def synthmem(p, h_next, i): 

    i *= 0

    hn1 = T.nnet.relu(ln(T.dot(h_next, p['Wh'][i]) + p['bh'][i]), alpha=0.02)
    hn2 = T.nnet.relu(T.dot(hn1, p['Wh2'][i]) + p['bh2'][i], alpha=0.02)

    xh1 = T.nnet.relu(ln(T.dot(h_next, p['Wx'][i]) + p['bx'][i]), alpha=0.02)
    x = T.dot(xh1, p['Wx2'][i]) + p['bx2'][i]
    
    yh1 = T.nnet.relu(ln(T.dot(h_next, p['Wy1'][i]) + p['by1'][i]), alpha=0.02)
    y = T.nnet.softmax(T.dot(yh1, p['Wy2'][i]))

    return hn2, x, y


params_forward = init_params_forward()
params_synthmem = init_params_synthmem()


'''
Set up the forward method and the synthmem_method
'''

x_true = T.ivector()
y_true = T.ivector()
h_in = T.matrix()
step = T.iscalar()

print "giving x and y on all steps"

y_true_use = y_true#T.switch(T.ge(step, 4), y_true, 10)

x_true_use = expand(x_true,n_feat).astype('float32')

h_next, y_est, class_loss,acc = forward(params_forward, h_in, x_true_use, y_true_use,step)

h_in_rec, x_rec, y_rec = synthmem(params_synthmem, h_next,step)

print "0.1 mult"
rec_loss = 0.0 * (T.sqr(x_rec - x_true_use).sum() + T.sqr(h_in - h_in_rec).sum() + crossent(y_rec, y_true_use))

#should pull y_rec and y_true together!  

print "TURNED OFF CLASS LOSS IN FORWARD"
#TODO: add in back params_forward.values()
updates_forward = lasagne.updates.adam(rec_loss + 1.0 * class_loss, params_forward.values() + params_synthmem.values(), learning_rate = 0.0001)

forward_method = theano.function(inputs = [x_true,y_true,h_in,step], outputs = [h_next, rec_loss, class_loss,acc,y_est], updates=updates_forward)
forward_method_noupdate = theano.function(inputs = [x_true,y_true,h_in,step], outputs = [h_next, rec_loss, class_loss,acc])


'''
Goal: get a method that takes h[i+1] and dL/dh[i+1].  It runs synthmem on h[i+1] to get estimates of x[i], y[i], and h[i].  It then runs the forward on those values and gets that loss.  


'''

h_next = T.matrix()
g_next = T.matrix()

h_last, x_last, y_last = synthmem(params_synthmem, h_next,step)

x_last = x_last
y_last = y_last.argmax(axis=1)

h_next_rec, y_est, class_loss,acc = forward(params_forward, h_last, x_last, y_last,step)

print "ONLY USING CLASS LOSS ON FINAL STEP"
class_loss = class_loss * T.eq(step,num_steps-1)

#g_next_use = g_next*1.0


print "DOING MATCHING TRICK"
g_next_use = g_next * T.eq(T.sgn(h_next), T.sgn(h_next_rec))

hdiff = T.eq(T.sgn(h_next), T.sgn(h_next_rec)).mean()
g_last = T.grad(class_loss, h_last, known_grads = {h_next_rec*1.0 : g_next_use})
g_last_local = T.grad(class_loss, h_last)


print "synthmem mult 1"
param_grads = T.grad(class_loss * 1.0, params_forward.values(), known_grads = {h_next_rec*1.0 : g_next_use})

#Should we also update gradients through the synthmem module?
synthmem_updates = lasagne.updates.adam(param_grads, params_forward.values(), learning_rate = 0.0001)

synthmem_method = theano.function(inputs = [h_next, g_next, step], outputs = [h_last, g_last, hdiff, g_last_local], updates = synthmem_updates)


'''
BPTT method: 
    -Run forward pass for many steps.  
    -Optimize total loss wrt all params.  
'''

x = T.imatrix()
y = T.imatrix()
step = T.iscalar()

def one_step(xval,yval,total_loss,total_acc,hval,yel):
    h_next,y_est,loss,acc = forward(params_forward, hval, expand(xval,2).astype('float32'),yval,step)
    
    return [(loss).astype('float32'),(acc).astype('float32'),h_next.astype('float32'),y_est.argmax(axis=1).astype('float32')]

h_next = theano.shared(0.0 * rng.normal(size = (64,1024)).astype('float32'))

y_est = theano.shared(np.zeros(shape=(64,)).astype('float32'))

scan_res, scan_updates = theano.scan(fn=one_step, sequences=[x.T,y.T], outputs_info=[np.asarray(0.0,dtype='float32'),np.asarray(0.0,dtype='float32'),h_next,y_est])

total_loss, total_acc, h_final, yest = scan_res

lr = 0.0001
print "learning rate", lr
updates = lasagne.updates.rmsprop(total_loss.mean(), params_forward.values(), learning_rate=lr)

t0 = time.time()

train_bptt = theano.function(inputs = [x,y,step], outputs = [total_loss.sum()/64.0, total_acc.mean(),yest], updates = updates)
print "time to compile", time.time() - t0

m = 1024

ta = []
tc = []
hlst = []

for iteration in xrange(0,100000):

    x,y = get_batch("train")

    t0 = time.time()
    
    loss,acc,yest = train_bptt(x,y,np.array(0.0).astype('int32'))

    print "yest shape", yest.shape

    print loss,acc

    if iteration % 100 == 0:
        print time.time() - t0, "TIME TO TRAIN ONE EXAMPLE"

        print "true", y[0,:].tolist()
        print "est", yest[:,0].tolist()





