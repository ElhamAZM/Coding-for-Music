import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten

data_path = './dataset/'
mfcc_path = './mfcc/'

MFCC_DIM = 10

def mean_mfcc(dataset='train'):
    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        mfcc_mat = np.zeros(shape=(MFCC_DIM, 1000))
    else:
        mfcc_mat = np.zeros(shape=(MFCC_DIM, 200))

    i = 0
    for file_name in f:
 
        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)
      

        # mean pooling
        temp = np.mean(mfcc, axis=1)
        mfcc_mat[:,i]= np.var(mfcc, axis=1)**1/2*np.mean(mfcc, axis=1)/2 #using variance and mean
        i = i + 1

    f.close();

    return mfcc_mat

   
    #book = np.array((whitened[1],whitened[2],whitened[3],whitened[4],whitened[5],whitened[6],whitened[7],whitened[8],whitened[9],whitened[10]))
        
    #mfcc = kmeans(whitened,20)
#whitened = whiten(mfcc_mat)
#mfcc = kmeans(whitened,10)
        
if __name__ == '__main__':
    train_data = mean_mfcc('train')
    valid_data = mean_mfcc('valid')
    test_data = mean_mfcc('test')

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,3)
    plt.imshow(test_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.show()



