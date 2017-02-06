#!/usr/bin/env python

import sys
sys.path.append("/data/lisatmp3/kenan/dreamprop/lib")

import cPickle as pickle
import gzip
from loss import accuracy, crossent, expand
import theano
import theano.tensor as T
import numpy.random as rng
import lasagne
import numpy as np
from random import randint

from load_cifar import CifarData

from discriminator import Discriminator

dataset = "mnist"
# dataset = "cifar"
#dataset = "ptb_char"

print "dataset", dataset

if dataset == "mnist":
    mn = gzip.open("/data/lisatmp3/kenan/data/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train
    validx,validy = valid

    trainy = trainy.astype('int32')
    validy = validy.astype('int32')

    nf = 784

elif dataset == "cifar":

    nf = 32*32*3

    config = {}
    config["cifar_location"] = "/data/lisatmp3/kenan/data/cifar-10-batches-py/"
    config['mb_size'] = 64
    config['image_width'] = 32

    cd_train = CifarData(config, segment="train")
    trainx = cd_train.images.reshape(50000,32*32*3) / 128.0 - 1.0
    trainy = cd_train.labels.astype('int32')
    cd_valid = CifarData(config, segment="test")
    validx = cd_valid.images.reshape(10000,32*32*3) / 128.0 - 1.0
    validy = cd_valid.labels.astype('int32')

    trainx = trainx.astype('float32')
    validx = validx.astype('float32')

print "train x", trainx

srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

print trainx.shape
print trainy.shape

print validx.shape
print validy.shape

num_steps = 5

print "doing deep bp"
print "using 1 layer forward net"
print "Number of steps", num_steps

def init_params_forward():

    p = {}

    p['W1'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024+nf,1024)).astype('float32'))
    #p['W2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['Wy'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024,11)).astype('float32'))
    #p['Wo'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024,1024)).astype('float32'))


    return p

def init_params_synthmem():

    p = {}

    p['Wh'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024,2048)).astype('float32'))
    p['Wh2'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,2048,1024)).astype('float32'))

    p['Wx'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024,2048)).astype('float32'))
    p['Wx2'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,2048,nf)).astype('float32'))

    p['Wy1'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024,1024)).astype('float32'))
    p['Wy2'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024,11)).astype('float32'))

    p['bh'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,2048,)).astype('float32'))
    p['bh2'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024,)).astype('float32'))
    
    p['bx'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,2048,)).astype('float32'))
    p['bx2'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,nf,)).astype('float32'))

    p['by1'] = theano.shared(0.03 * rng.normal(0,1,size=(num_steps,1024,)).astype('float32'))

    return p


def join2(a,b):
        return T.concatenate([a,b], axis = 1)

def ln(inp):
    return (inp - T.mean(inp,axis=1,keepdims=True)) / (0.001 + T.std(inp,axis=1,keepdims=True))


def forward(p, h, x_true, y_true, i):

    inp = join2(h, x_true)

    h1 = T.nnet.relu(ln(T.dot(inp, p['W1'][i])), alpha=0.02)
    #h2 = T.nnet.relu(ln(T.dot(h1, p['W2'])), alpha=0.02)
    h2 = h1

    y_est = T.nnet.softmax(T.dot(h2, p['Wy'][i]))

    #h_next = T.dot(h2, p['Wo'][i])
    h_next = h1

    loss = crossent(y_est, y_true)

    acc = accuracy(y_est, y_true)

    return h_next, y_est, loss, acc

def synthmem(p, h_next, i): 

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

x_true = T.matrix()
y_true = T.ivector()
h_in = T.matrix()
step = T.iscalar()

print "giving x and y on all steps"

y_true_use = y_true

x_true_use = x_true

h_next, y_est, class_loss,acc = forward(params_forward, h_in, x_true_use, y_true_use,step)

h_in_rec, x_rec, y_rec = synthmem(params_synthmem, h_next,step)

print "0.1 mult"
# rec_loss = 0.1 * (T.sqr(x_rec - x_true_use).sum() + T.sqr(h_in - h_in_rec).sum() + crossent(y_rec, y_true_use))

# replace square loss by GAN loss
g_cost_aux = T.sqr(x_rec - x_true_use).sum()
d_cost, g_cost_1, d_params = Discriminator(x_true_use, x_rec ) 
g_cost_2 = T.sqr(h_in - h_in_rec).sum()
#d_cost_2, g_cost_2, d_params = Discriminator(x_true_use, x_rec)
g_cost_3 = crossent(y_rec, y_true_use)
g_cost = ( g_cost_1 + g_cost_2 +  g_cost_3)
d_cost = d_cost
rec_loss = 0.1 * g_cost

#should pull y_rec and y_true together!  

print "TURNED OFF CLASS LOSS IN FORWARD"
#TODO: add in back params_forward.values()
updates_forward = lasagne.updates.adam(rec_loss + 0.0 * class_loss, params_forward.values() + params_synthmem.values() )
updates_d = lasagne.updates.sgd(d_cost, d_params, 0.1)
updates_forward.update(updates_d)

forward_method = theano.function(inputs = [x_true,y_true,h_in,step], outputs = [h_next, rec_loss, g_cost_aux, d_cost, class_loss,acc,y_est], updates=updates_forward)
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

#out_grad = T.grad(loss, h_in_init, known_grads = {h_out*1.0 : in_grad * T.gt(h_in_init,0.0)})
#param_grads = T.grad(net['loss'], params.values(), known_grads = {net['h_out']*1.0 : in_grad*1.0})

print "synthmem mult 1"
param_grads = T.grad(class_loss * 1.0, params_forward.values(), known_grads = {h_next_rec*1.0 : g_next_use})

#Should we also update gradients through the synthmem module?
synthmem_updates = lasagne.updates.adam(param_grads, params_forward.values())

synthmem_method = theano.function(inputs = [h_next, g_next, step], outputs = [h_last, g_last, hdiff, g_last_local], updates = synthmem_updates)

m = 1024

for iteration in xrange(0,100000):
    r = randint(0,49900)
    x = trainx[r:r+64]
    y = trainy[r:r+64]

    h_in = np.zeros(shape=(64,m)).astype('float32')

    for j in range(num_steps):
        h_next, rec_loss, g_cost_aux, d_cost, class_loss,acc,y_est = forward_method(x,y,h_in,j)
        h_in = h_next
        #print "est", y_est
        #print "true", y

    g_next = np.zeros(shape=(64,m)).astype('float32')


    for k in reversed(range(0,num_steps)):
        h_next, g_last,hdiff,g_last_local = synthmem_method(h_next,g_next,k)
        g_next = g_last

    #using 500
    if iteration % 100 == 0:
        print "========================================"
        print "train acc", acc
        print "train cost", class_loss
        print "train rec_loss", rec_loss
        print "discriminator loss", d_cost
        print 'g_cost_aux', g_cost_aux
        va = []
        vc = []
        for ind in range(0,10000,1000):
            h_in = np.zeros(shape=(1000,m)).astype('float32')
            for j in range(num_steps):
                h_next,rec_loss,class_loss,acc = forward_method_noupdate(validx[ind:ind+1000], validy[ind:ind+1000], h_in, j)
                h_in = h_next

            va.append(acc)
            vc.append(class_loss)

        print "REVERSED RANGE"
        print "Iteration", iteration
        print "Valid accuracy", sum(va)/len(va)
        print "Valid cost", sum(vc)/len(vc)





