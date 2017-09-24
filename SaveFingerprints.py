# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:53:11 2017

@author: wuyiming
"""

import numpy as np
import os
import sys
from librosa.util import find_files
from librosa.core import load
import analyze
import pickle

def hash_to_path(h):
    h_str = "%05x" % h
    directory = os.path.join("hashtable",h_str[:3])
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.join(directory,h_str+".npz")

args = sys.argv

audiolist = find_files("/home/wuyiming/Projects/ChordData/Audio","wav")

hash_table = {}
title_list = []

audio_id = 0
for audiofile in audiolist:
    print ("analyzing %03d: " % audio_id)  + audiofile
    y,sr = load(audiofile,sr=11025)
    title_list.append(audiofile.split("/")[-1])
    list_landmarks = analyze.extract_landmarks(y)
    for landmark in list_landmarks:
        hsh = landmark[0]
        starttime = landmark[1]
        if hash_table.has_key(hsh):
            hash_table[hsh].append((audio_id,starttime))
        else:
            hash_table[hsh] = [(audio_id,starttime)]
    
    audio_id += 1
    

print "Writing..."
with open("hashtable.pkl",mode="wb") as f:
    pickle.dump(hash_table,f)
    
with open("titlelist.pkl",mode="wb") as f:
    pickle.dump(title_list,f)