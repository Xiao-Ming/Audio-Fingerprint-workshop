# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 18:40:04 2017

@author: wuyiming
"""

import analyze
from librosa import load
import matplotlib.pyplot as plt
import numpy as np
import Database as D



y_query,sr = load("query.wav",sr=11025)
landmarks_query = analyze.extract_landmarks(y_query)
database = D.Database()

plt.subplot(3,1,1)
database.plot_match(y_query,119)
plt.title("Back In The U.S.S.R")


plt.subplot(3,1,2)
database.plot_match(y_query,163)
plt.title("Two Of Us")


plt.subplot(3,1,3)
database.plot_match(y_query,156)
plt.title("Sun King")
