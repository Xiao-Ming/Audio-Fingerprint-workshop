# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 21:10:36 2017

@author: wuyiming
"""

import numpy as np
import analyze
import pickle
import matplotlib.pyplot as plt

DIFF_RANGE = 2000000

class Database:
    def __init__(self):
        with open("hashtable.pkl","rb") as f:
            self.hash_table = pickle.load(f)
        with open("titlelist.pkl","rb") as f:
            self.title_list = np.array(pickle.load(f))
    
    def query_histogram(self,y,best=5):
        list_landmarks = analyze.extract_landmarks(y)
        id_candidates = self._find_candidate_id(list_landmarks)
        candidate_landmark_histogram = np.zeros(id_candidates.size,dtype=np.int32)
        for landmark in list_landmarks:
            hsh = landmark[0]
            if self.hash_table.has_key(hsh):
                hit_ids = [item[0] for item in self.hash_table[hsh]]
                for i in hit_ids:
                    candidate_landmark_histogram[np.where(id_candidates==i)[0]] += 1
            else:
                continue
        
        idxsort = np.argsort(candidate_landmark_histogram)[::-1]
        title_sort = self.title_list[id_candidates[idxsort]]
        fingerprint_histogram_sort = candidate_landmark_histogram[idxsort]
        return title_sort[:best],fingerprint_histogram_sort[:best]
        
    def query_diff_histogram(self,y,best=5):
        list_landmarks = analyze.extract_landmarks(y)
        id_candidates = self._find_candidate_id(list_landmarks)
        diff_histogram = np.zeros((id_candidates.size,DIFF_RANGE),dtype=np.int32)
        diff_start_offset = np.zeros(id_candidates.size,dtype=np.int32)
        
        for landmark in list_landmarks:
            hsh_query = landmark[0]
            offset_query = landmark[1]
            if self.hash_table.has_key(hsh_query):
                hits = self.hash_table[hsh_query]
                for hit in hits:
                    id_hit = hit[0]
                    offset_hit = hit[1]
                    diff = offset_hit - offset_query
                    if diff >= 0:
                        id_hist = np.where(id_candidates==id_hit)[0]
                        diff_histogram[id_hist,diff] += 1
                        diff_start_offset[id_hist] = min([diff_start_offset[id_hist],offset_query])
            else:
                continue
            
        candidate_match_score = np.max(diff_histogram,axis=1)
        candidate_max_offset = np.argmax(diff_histogram,axis=1)
        idxsort = np.argsort(candidate_match_score)[::-1]
        title_sort = self.title_list[id_candidates[idxsort]]
        match_score_sort = candidate_match_score[idxsort]
        best_match_offset = (candidate_max_offset[idxsort[0]] - diff_start_offset[idxsort[0]]) * 256/11025.0
        return title_sort[:best],match_score_sort[:best],best_match_offset
        
    def plot_match(self,y_query,target_id):
        list_landmarks = analyze.extract_landmarks(y_query)
        t_target = []
        t_query = []
        for landmark in list_landmarks:
            hsh_query = landmark[0]
            offset_query = landmark[1]
            if self.hash_table.has_key(hsh_query):
                hits = self.hash_table[hsh_query]
                for hit in hits:
                    id_hit = hit[0]
                    offset_hit = hit[1]
                    if id_hit == target_id:
                        t_target.append(offset_hit)
                        t_query.append(offset_query)
            else:
                continue
        
        plt.plot(np.array(t_target),np.array(t_query),"bo")
        
                   
    def _find_candidate_id(self,landmarks):
        candidate = set()
        for landmark in landmarks:
            hsh = landmark[0]
            if self.hash_table.has_key(hsh):
                id_hits = [item[0] for item in self.hash_table[hsh]]
                candidate = candidate.union(id_hits)
            else:
                continue
            
        return np.array(list(candidate),dtype=np.int32)