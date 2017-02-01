import theano
import theano.tensor as T
import numpy as np
#import numpy.random as rng

class HiddenLayer:

    def __init__(self, num_in, num_out, activation = 'lrelu', batch_norm = False, layer_norm = False, norm_prop = False):

        self.activation = activation
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.norm_prop = norm_prop

        self.residual = (num_in == num_out)

        self.W = theano.shared(0.01 * np.random.normal(size = (num_in, num_out)).astype('float32'))
        self.b = theano.shared(0.0 * np.random.normal(size = (num_out)).astype('float32'))

        if self.batch_norm or self.layer_norm or self.norm_prop:
            bn_mean_values = 1.0 * np.zeros(shape = (1, num_out)).astype('float32')
            bn_std_values = 1.0 * np.ones(shape = (1,num_out)).astype('float32')
            self.bn_mean = theano.shared(value=bn_mean_values)
            self.bn_std = theano.shared(value = bn_std_values)
            self.params = [self.bn_mean, self.bn_std]
        else:
            self.params = []


        self.params += [self.W, self.b]

    def output(self, input_raw):

        input = input_raw

        lin_output = T.dot(input, self.W) + self.b

        if self.batch_norm:
            lin_output = (lin_output - T.mean(lin_output, axis = 0, keepdims = True)) / (1.0 + T.std(lin_output, axis = 0, keepdims = True))
            lin_output = (lin_output * T.addbroadcast(self.bn_std,0) + T.addbroadcast(self.bn_mean,0))

        if self.layer_norm:
            lin_output = (lin_output - T.mean(lin_output, axis = 1, keepdims = True)) / (1.0 + T.std(lin_output, axis = 1, keepdims = True))
            lin_output = (lin_output * T.addbroadcast(self.bn_std,0) + T.addbroadcast(self.bn_mean,0))

        if self.norm_prop:
            lin_output = lin_output / T.sqrt(T.mean(T.sqr(lin_output), axis = 0))
            lin_output = (lin_output * T.addbroadcast(self.bn_std,0) + T.addbroadcast(self.bn_mean,0))

        clip_preactive = True

        if clip_preactive:
            lin_output = theano.tensor.clip(lin_output, -10, 10)
        
        self.out_store = lin_output


        if self.activation == None:
            activation = lambda x: x
        elif self.activation == "relu":
            activation = lambda x: T.maximum(0.0, x)
        elif self.activation == "lrelu":
            activation = lambda x: T.nnet.relu(x, alpha = 0.02)
        elif self.activation == "exp":
            activation = lambda x: T.exp(x)
        elif self.activation == "tanh":
            activation = lambda x: T.tanh(x)
        elif self.activation == 'softplus':
            activation = lambda x: T.nnet.softplus(x)
        elif self.activation == 'sigmoid':
            activation = lambda x: T.nnet.sigmoid(x)
        else:
            raise Exception("Activation not found")

        out = activation(lin_output)

        return out
