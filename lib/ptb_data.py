

'''
Code responsible for loading data for running on penn treebank

-Can load whole dataset into memory.  

-Divide into chunks based on minibatch indices.  I.e. N/64 sized chunks.  

-Have a read method that keeps reading until the end and loops over to beginning when it almost runs out.  

-Y should just be like X but set one ahead in the future.  

-Initially could just run dreamprop over a handful of steps.  Should still get somewhere reasonable.  

X: (ind, minibatch, time)

-For characters, compute indices in 100-dim space?  

'''

data = "/u/lambalex/Downloads/char_penntree.npz"

import numpy as np

fh = np.load(data)

train = fh['train']
valid = fh['valid']
test = fh['test']

train_len = train.shape[0] - train.shape[0] % 64

train_m = train[:train_len].reshape((64, train_len/64))

valid_len = valid.shape[0] - valid.shape[0] % 64

valid_m = valid[:valid_len].reshape((64, valid_len/64))

print train_m.shape
print valid_m.shape

'''
79000, train
6100, valid
'''
def get_batch(segment,index):
    if segment == "train":
        return train_m[:,index], train_m[:,index+1]
    else:
        return valid_m[:,index], valid_m[:,index+1]

#0 to 49 is index range.  

#First break each into chunks of length N/64.  

if __name__ == "__main__":

    for ind in range(0,100):
        x,y = get_batch("train", ind)

        print "x", x, "y", y



