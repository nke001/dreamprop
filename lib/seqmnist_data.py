'''
Code for running on sequential MNIST.  
'''
import numpy as np

datafile = "/u/lambalex/data/binarized_mnist/structured/train.txt"

fh = open(datafile, "r")

train = []

for line in fh:
    train += [np.asarray(map(int,(line[:-1])))]

train = np.array(train)[::4]

print train.shape

def get_batch(segment, index):
    return train[0:64,index].astype('int32'), train[0:64, index+1].astype('int32')

if __name__ == "__main__":
    fh = open(datafile,"r")

    for line in fh:
        print line

