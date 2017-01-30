'''
Code for running on sequential MNIST.  
'''
import numpy as np

datafile = "/u/lambalex/data/binarized_mnist/structured/train.txt"

fh = open(datafile, "r")

train = []

for line in fh:
    train += [np.asarray(map(int,(line[:-1])))]

train = np.array(train)

print train.shape

import random

def get_batch(segment):
    r = random.randint(0,49900)
    return train[r:r+64,:-1].astype('int32'), train[r:r+64,1:].astype('int32')

if __name__ == "__main__":

    x,y = get_batch('train')

    print x.shape
    print y.shape

    print x[0,300:350]
    print y[0,300:350]


