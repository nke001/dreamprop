'''
Code for running on sequential MNIST.  
'''
import numpy as np

datafile = "/data/lisatmp3/kenan/data/binarized_mnist/structured/train.txt"

fh = open(datafile, "r")

train = []

for line in fh:
    train += [np.asarray(map(int,(line[:-1])))]

train = np.array(train)

print train.shape

datafile = "/u/lambalex/data/binarized_mnist/structured/test.txt"

fh = open(datafile, "r")

test = []

for line in fh:
    test += [np.asarray(map(int,(line[:-1])))]

test = np.array(test)

print test.shape

import random

def get_batch(segment,mb=64):

    if segment == "train":
        r = random.randint(0,49400)
        return train[r:r+mb,:-1].astype('int32'), train[r:r+mb,1:].astype('int32')
    elif segment == "test":
        r = random.randint(0,9400)
        return test[r:r+mb,:-1].astype('int32'), test[r:r+mb,1:].astype('int32')

if __name__ == "__main__":

    x,y = get_batch('train')

    print x.shape
    print y.shape

    print x[0,300:350]
    print y[0,300:350]


