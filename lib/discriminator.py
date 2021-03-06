import theano 
import theano.tensor as T

from theano.tensor.nnet import binary_crossentropy
from hidden import HiddenLayer

srng = theano.tensor.shared_randomstreams.RandomStreams(320)

def layers2params(layers):
    paramList = []

    for layer in layers:
        paramList += layer.params

        print 'layer ', layer
        print 'params ', layer.params

    return paramList

def Discriminator(real, fake):
    in_len = 784
    m_disc = 512

    batch_size = 64
    
    pair = T.concatenate([real, fake], axis=0)

    h1 = HiddenLayer(in_len, m_disc)
    h2 = HiddenLayer(m_disc, m_disc)
    h3 = HiddenLayer(m_disc, 1, activation = 'sigmoid')

    pc1 = h1.output(pair)
    pc2 = h2.output(pc1)
    pc3 = h3.output(pc2)

    p_real = pc3[:real.shape[0],: ].flatten()
    p_gen = pc3[ -real.shape[0]: , :].flatten()


    d_cost_real = binary_crossentropy(p_real, T.ones(p_real.shape)).mean()
    d_cost_real = (d_cost_real * (d_cost_real < 0.9)).mean()
    d_cost_gen = binary_crossentropy(p_gen, T.zeros(p_gen.shape)).mean()
    d_cost_gen = (d_cost_gen * (d_cost_gen > 0.1)).mean()
    g_cost_d = binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean()
    
    d_cost =  (d_cost_real + d_cost_gen ) / 2.0
    g_cost = g_cost_d
    
    layers = [h1,h2,h3]
    params = layers2params(layers)
    
    return d_cost, g_cost, params



