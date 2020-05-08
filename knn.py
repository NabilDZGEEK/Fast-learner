# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:58:24 2020

@author: nabil
"""
import numpy as np
import preprocessing as prepro

k=0
data=[]
label=[]

def evaluate(data,label):
    pr=predict(data)
    taille=len(label)
    count=0
    for i in range(taille):
        if pr[i]==label[i]:
            count+=1      
    return count*1.0/taille*100     
def distance(x,y):
  squared_difference = 0
  for i in range(len(x)):
    squared_difference += (x[i] - y[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance
  

def fit(d,l,kn):
     global data
     global label
     global k
     
     data=prepro.normalize(d)
     label=l
     k=kn
  
def predict(p): 
    def min_distance(x):
        labels,counts=np.unique(x[:,1],return_counts=True)
        valmax=  counts.max()      
        ind=np.where(counts==valmax)[0][0]
        return labels[ind]       
    result=[]
    for row in p:
        distances=[]
        for i in range(len(data)):
            distances.append([distance(row,data[i]),label[i]])    
        distances.sort()     
        k_neighbors=np.array(distances[:k])
        result.append(min_distance(k_neighbors))
    return result