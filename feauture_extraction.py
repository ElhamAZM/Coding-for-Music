import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import math
from scipy.cluster.vq import vq, kmeans, whiten

data_path = './dataset/'
mfcc_path = './mfcc/'

MFCC_DIM = 10 # reduce the size of texture window 

def extract_mfcc(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print (i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        #print file_path
        y, sr = librosa.load(file_path, sr=22050)


        y= librosa.core.to_mono(y) # Force an audio signal down to mono
        
        for j in range(1,88200):
            y[j]=y[j]*2*abs(math.cos(math.pi*j/88200)) # manual normalization, (giving a fade in and fade out)

        y=librosa.util.normalize(y) 
        

         # STFT
        S = librosa.core.stft(y, n_fft=2048, hop_length=256, win_length=2048)# 2048 for best resulotion

        #power spectrum
        D = np.abs(S)**1
         #mel spectrogram (512 --> 40)
        mel_basis = librosa.filters.mel(sr, 2048, n_mels=40 )# 40 frequency bank (notes)
        
        mel_S = np.dot(mel_basis, D)
        #log compression
        
        log_mel_S = librosa.power_to_db(mel_S)
         #mfcc (DCT)
        mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=10) # devided to 10 cluster suitable for 10 instrument
        mfcc = mfcc.astype(np.float32) # to save the memory (64 to 32 bits)


        # save mfcc as a file
        file_name = file_name.replace('.wav','.npy')
        save_file = mfcc_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc)
        

    f.close();
if __name__ == '__main__':
    extract_mfcc(dataset='train')                 
    extract_mfcc(dataset='valid')                                  
    extract_mfcc(dataset='test')
    
