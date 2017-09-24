# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:44:57 2017

@author: wuyiming
"""


import numpy as np
import scipy.ndimage as ndi
from librosa.core import stft


F1_BITS = 8
DF_BITS = 6
DT_BITS = 6

B1_MASK = (1<<F1_BITS) - 1
B1_SHIFT = DF_BITS + DT_BITS
DF_MASK = (1<<DF_BITS) - 1
DF_SHIFT = DT_BITS
DT_MASK = (1<<DT_BITS) - 1

def extract_landmarks(y,peak_dist=20):
    peaks_mat = find_peaks(y,peak_dist)
    landmarks = peaks_to_landmarks(peaks_mat)
    return landmarks

def find_peaks(y,size):
    sgram = np.abs(stft(y,n_fft=512,hop_length=256))
    #sgram = np.log(np.maximum(sgram,np.max(sgram)/1e6))
    #sgram = sgram - np.mean(sgram)
    sgram_max = ndi.maximum_filter(sgram,size=size,mode="constant")
    maxima = (sgram==sgram_max) & (sgram > 0.2)
    return maxima
    
    
def peaks_to_landmarks(peaks_mat,target_dist=30,target_time=30,target_freq=30):
    peaks_freq,peaks_time = np.where(peaks_mat)
    list_landmarks = []
    for pfreq,ptime in zip(peaks_freq,peaks_time):
        #(px,py) -- current anchor
        target_mask = np.zeros(peaks_mat.shape,dtype=np.bool)
        target_mask[pfreq-target_freq:pfreq+target_freq,ptime+target_dist:ptime+target_dist+target_time] = 1
        targets_freq,targets_time = np.where(peaks_mat & target_mask)
        for pfreq_target,ptime_target in zip(targets_freq,targets_time):
            dtime = ptime_target - ptime
            hsh = (((pfreq & B1_MASK)<<B1_SHIFT) | (((pfreq_target+target_freq-pfreq)&DF_MASK)<<DF_SHIFT)|(dtime&DT_MASK))
            list_landmarks.append((hsh,ptime))
        
        
    return list_landmarks
    