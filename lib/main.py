#!/usr/bin/env python

import sys
sys.path.append("/u/lambalex/DeepLearning/dreamprop/lib")

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


mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

train, valid, test = pickle.load(mn)

trainx,trainy = train
validx,validy = valid

trainy = trainy.astype('int32')
validy = validy.astype('int32')

#config = {}
#config["cifar_location"] = "/u/lambalex/data/cifar/cifar-10-batches-py/"
#config['mb_size'] = 128
#config['image_width'] = 32

#cd_train = CifarData(config, segment="train")
#trainx = cd_train.images.reshape(50000,32*32*3) / 128.0 - 1.0
#trainy = cd_train.labels
#cd_valid = CifarData(config, segment="test")
#validx = cd_valid.images.reshape(10000,32*32*3) / 128.0 - 1.0
#validy = cd_valid.labels

print "train x", trainx

srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

print trainx.shape
print trainy.shape

print validx.shape
print validy.shape

num_steps = 1

print "Number of steps", num_steps

def init_params_forward():

    p = {}

    p['W1'] = theano.shared(0.03 * rng.normal(0,1,size=(1024+784,1024)).astype('float32'))
    p['W2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['Wy'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,10)).astype('float32'))
    p['Wo'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))


    return p

def init_params_synthmem():

    p = {}

    p['W1'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['W2'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['Wh'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,1024)).astype('float32'))
    p['Wx'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,784)).astype('float32'))
    p['Wy'] = theano.shared(0.03 * rng.normal(0,1,size=(1024,10)).astype('float32'))

    return p


def join2(a,b):
        return T.concatenate([a,b], axis = 1)

def ln(inp):
    return (inp - T.mean(inp,axis=1,keepdims=True)) / (0.001 + T.std(inp,axis=1,keepdims=True))


def forward(p, h, x_true, y_true):

    inp = join2(h, x_true)

    h1 = T.nnet.relu(ln(T.dot(inp, p['W1'])), alpha=0.02)
    h2 = T.nnet.relu(ln(T.dot(h1, p['W2'])), alpha=0.02)

    y_est = T.nnet.softmax(T.dot(h2, p['Wy']))

    h_next = T.dot(h2, p['Wo'])

    loss = crossent(y_est, y_true)

    acc = accuracy(y_est, y_true)

    return h_next, y_est, loss, acc

def synthmem(p, h_next): 

    h1 = T.nnet.relu(T.dot(h_next, p['W1']), alpha=0.02)
    h2 = T.nnet.relu(T.dot(h1, p['W2']), alpha=0.02)

    h = T.dot(h2, p['Wh'])
    x = T.dot(h2, p['Wx'])
    y = T.nnet.softmax(T.dot(h2, p['Wy']))

    return h, x, y




params_forward = init_params_forward()
params_synthmem = init_params_synthmem()


'''
Set up the forward method and the synthmem_method
'''

x_true = T.matrix()
y_true = T.ivector()
h_in = T.matrix()

h_next, y_est, class_loss,acc = forward(params_forward, h_in, x_true, y_true)

h_in_rec, x_rec, y_rec = synthmem(params_synthmem, h_next)

rec_loss = 1.0 * (T.abs_(x_rec - x_true).mean() + T.abs_(h_in - h_in_rec).mean() + T.sqr(expand(y_true) - y_rec).mean())

#should pull y_rec and y_true together!  

#NOTE NOTE NOTE: TRIED TURNING OFF CLASS LOSS IN FORWARD
print "TURNED OFF CLASS LOSS IN FORWARD"
updates_forward = lasagne.updates.adam(rec_loss + 0.0 * class_loss, params_forward.values() + params_synthmem.values())

forward_method = theano.function(inputs = [x_true,y_true,h_in], outputs = [h_next, rec_loss, class_loss,acc,y_est], updates=updates_forward)
forward_method_noupdate = theano.function(inputs = [x_true,y_true,h_in], outputs = [h_next, rec_loss, class_loss,acc])



'''
Goal: get a method that takes h[i+1] and dL/dh[i+1].  It runs synthmem on h[i+1] to get estimates of x[i], y[i], and h[i].  It then runs the forward on those values and gets that loss.  


'''

h_next = T.matrix()
g_next = T.matrix()

h_last, x_last, y_last = synthmem(params_synthmem, h_next)

x_last = 1.0*x_last + 0.0 * x_true
y_last = 0*y_last.argmax(axis=1) + y_true

h_next_rec, y_est, class_loss,acc = forward(params_forward, h_last, x_last, y_last)

#g_next_use = g_next*1.0

rec_weight = 1.0 / (1.0 + T.abs_(h_next -  h_next_rec).mean())
g_next_use = g_next * rec_weight

g_last = T.grad(class_loss, h_last, known_grads = {h_next_rec*1.0 : g_next_use})

#out_grad = T.grad(loss, h_in_init, known_grads = {h_out*1.0 : in_grad * T.gt(h_in_init,0.0)})
#param_grads = T.grad(net['loss'], params.values(), known_grads = {net['h_out']*1.0 : in_grad*1.0})

param_grads = T.grad(class_loss, params_forward.values(), known_grads = {h_next_rec*1.0 : g_next_use})

#Should we also update gradients through the synthmem module?
synthmem_updates = lasagne.updates.adam(param_grads, params_forward.values())

synthmem_method = theano.function(inputs = [h_next, g_next, x_true, y_true], outputs = [h_last, g_last, rec_weight, g_last], updates = synthmem_updates)


m = 1024

for iteration in xrange(0,5000):
    r = randint(0,49900)
    x = trainx[r:r+64]
    y = trainy[r:r+64]

    h_in = np.zeros(shape=(64,m)).astype('float32')

    for j in range(num_steps):
        h_next, rec_loss, class_loss,acc,y_est = forward_method(x,y,h_in)
        h_in = h_next
        #print "est", y_est
        #print "true", y

    g_next = np.zeros(shape=(64,m)).astype('float32')

    for k in range(0,1):
        h_next, g_next, rec_weight,g_last = synthmem_method(h_next,g_next,x,y)
        #print k, "REC WEIGHT", rec_weight

    #using 500
    if iteration % 5 == 0:
        print "train acc", acc
        print "train cost", class_loss
        print "train rec_loss", rec_loss
        va = []
        vc = []
        for ind in range(0,10000,1000):
            h_in = np.zeros(shape=(1000,m)).astype('float32')
            for j in range(num_steps):
                h_next,rec_loss,class_loss,acc = forward_method_noupdate(validx[ind:ind+1000], validy[ind:ind+1000], h_in)
                h_in = h_next

            va.append(acc)
            vc.append(class_loss)

        print "Iteration", iteration
        print "Valid accuracy", sum(va)/len(va)
        print "Valid cost", sum(vc)/len(vc)





