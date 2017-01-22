
import cPickle as pickle
import gzip
from loss import accuracy, crossent
import theano
import theano.tensor as T
import numpy.random as rng
import lasagne
import numpy as np
from random import randint

from load_cifar import CifarData

#mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

#train, valid, test = pickle.load(mn)

#trainx,trainy = train
#validx,validy = valid

#trainy = trainy.astype('int32')
#validy = validy.astype('int32')

config = {}

config["cifar_location"] = "/u/lambalex/data/cifar/cifar-10-batches-py/"
config['mb_size'] = 128
config['image_width'] = 32

cd_train = CifarData(config, segment="train")
trainx = cd_train.images.reshape(50000,32*32*3) / 128.0 - 1.0
trainy = cd_train.labels
cd_valid = CifarData(config, segment="test")
validx = cd_valid.images.reshape(10000,32*32*3) / 128.0 - 1.0
validy = cd_valid.labels

print "train x", trainx

srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

print trainx.shape
print trainy.shape

print validx.shape
print validy.shape

m = 512
num_internal_steps = 1
num_steps = 10

def init_params():

    params = {}

    i = 32*32*3

    params["A_W"] = theano.shared(0.01 * rng.normal(size = (num_steps, m+i,m)).astype('float32'))
    params["B_W"] = theano.shared(0.01 * rng.normal(size = (num_steps, m+i,m)).astype('float32'))
    params["C_W"] = theano.shared(0.01 * rng.normal(size = (num_steps, m,10)).astype('float32'))

    params["A_b"] = theano.shared(0.0 * rng.normal(size = (num_steps, m,)).astype('float32'))
    params["B_b"] = theano.shared(0.0 * rng.normal(size = (num_steps, m,)).astype('float32'))
    params["C_b"] = theano.shared(0.0 * rng.normal(size = (num_steps, 10,)).astype('float32'))

    return params

def join(a,b):
    return T.concatenate([a,b], axis = 1)

def bn(inp):
    return inp#(inp - T.mean(inp,axis=1,keepdims=True)) / (0.001 + T.std(inp,axis=1,keepdims=True))

def network(x_in, y, h_in, p, in_grad, t):

    relu = T.nnet.relu
    h_in_init = h_in

    loss = []
    acc = []
    problst = []

    for step in range(num_internal_steps):

        x = x_in#*T.cast(srng.binomial(n=1,p=0.005,size=x_in.shape), 'float32')

        delta_1 = relu(bn(T.dot(join(h_in,x), p['B_W'][t]) + p['B_b'][t]))
        delta_2 = bn(T.dot(join(delta_1,x), p['A_W'][t]) + p['A_b'][t])

        h_out = relu(delta_2 + h_in)

        h_out = h_out*0.0 + relu(bn(p['B_b'][t] + T.dot(join(h_in,x), p['B_W'][t])))

        prob = T.nnet.softmax(T.dot(h_out, p['C_W'][t]) + p['C_b'][t])

        h_in = h_out

        problst += [prob[0:1]]
        loss += [crossent(prob, y)]
        acc += [accuracy(prob, y)]

    loss = sum(loss) / len(loss)
    acc = sum(acc) / len(acc)

    out_grad = T.grad(loss, h_in_init, known_grads = {h_out*1.0 : in_grad * T.gt(h_in_init,0.0)})

    return {'loss' : loss, 'acc' : acc, 'h_out' : h_out, 'out_grad' : out_grad, 'prob' : T.stack(problst)}

params = init_params()
x = T.matrix()
y = T.ivector()
h_in = T.matrix()
in_grad = T.matrix()
t = T.iscalar()

net = network(x,y,h_in,params, in_grad, t)

param_grads = T.grad(net['loss'], params.values(), known_grads = {net['h_out']*1.0 : in_grad*1.0})

updates = lasagne.updates.adam(param_grads, params.values())

train_method = theano.function(inputs = [x,y,h_in,in_grad,t], outputs = {'loss' : net['loss'], 'acc' : net['acc'], 'h_out' : net['h_out'], 'out_grad' : net['out_grad'], 'prob' : net['prob']}, updates = updates,allow_input_downcast=True)
valid_method = theano.function(inputs = [x,y,h_in,t], outputs = {'loss' : net['loss'], 'acc' : net['acc'], 'h_out' : net['h_out']}, allow_input_downcast=True)

for iteration in xrange(0,500000):
    r = randint(0,49900)
    x = trainx[r:r+64]
    y = trainy[r:r+64]

    h_in = np.zeros(shape=(64,m)).astype('float32')
    in_grad = np.zeros(shape=(64,m)).astype('float32')

    for j in range(num_steps):
        out = train_method(x,y,h_in,in_grad,j)
        h_in = out['h_out']
        in_grad = out['out_grad']
        if j == 0:
            in_grad *= 0.0
        if iteration % 100 == 0:
            print "grads", j, in_grad.mean()
            #print "prob", j, out['prob'][0].round(2)
            #print "h val", j, h_in.round(2)

    #using 500
    if iteration % 100 == 0:
        print "train acc", out['acc']
        print "train cost", out['loss']
        va = []
        vc = []
        for ind in range(0,10000,1000):
            h_in = np.zeros(shape=(1000,m)).astype('float32')
            for j in range(num_steps):
                valid_out = valid_method(validx[ind:ind+1000], validy[ind:ind+1000], h_in, j)
                h_in = valid_out['h_out']

            va.append(valid_out['acc'])
            vc.append(valid_out['loss'])

        print "Iteration", iteration
        print "Valid accuracy", sum(va)/len(va)
        print "Valid cost", sum(vc)/len(vc)



