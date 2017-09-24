# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 22:38:13 2017

@author: wuyiming
"""

import numpy as np
import sys
from librosa.core import load
from librosa.util import find_files
import analyze
import Database
import time
import matplotlib.pyplot as plt

args = sys.argv

#query_file = args[1]
query_type = "diff"

print "loading database..."
database = Database.Database()


print "querying..."
t = time.time()
y_query,sr = load("query.wav",sr=11025)

if query_type == "hist":
    result_title_list,result_score_list = database.query_histogram(y_query)
elif query_type == "diff":
    result_title_list,result_score_list,best_match_offset = database.query_diff_histogram(y_query)
else:
    print "Please select valid query type!"
    exit()


print "TITLE\tSCORE"
for title,score in zip(result_title_list,result_score_list):
    print "%s\t%d" % (title,score)

if query_type == "diff":
    print "matched offset: %.3f sec." % best_match_offset

print "time: %.3f sec." % (time.time() - t)

plt.bar(np.arange(5),result_score_list,tick_label=[s for s in result_title_list],align="center")
#plt.tight_layout()