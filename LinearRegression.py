# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:40:55 2020

@author: nabil
"""

import numpy as np
import pandas as pd

features=[]
def mse(predicted,labels):
    dif=predicted-labels
    dif=dif*dif
    return dif.sum()
def fit(data,labels):
    data=np.array(data)
    labels=np.array(labels)
    
    def moy(x,y):
        return np.array(data[:,x]*data[:,y]).mean()
    #get number of columns
    d=data.shape[1]
    #A*X=Y ==> X=inverse(A)*Y
    #creating  matrix y
    y=[np.array(data[:,i]*labels).mean() for i in range(d)]
    y.append(labels.mean())
    extrem=[data[:,i].mean() for i in range(d)]+[1]
    #creating  matrix A
    a=[]
    for i in range(d):
        row=[moy(i,j) for j in range (d)]+[extrem[i]]
        a.append(row)
    a.append(extrem)
    a=np.array(a)
    inv=np.linalg.inv(a)
    global features
    features=np.dot(inv,y)

def predict(p):
    add=np.array([[1]]*len(p))
    p=np.append(p,add,axis=1)
    output=[]
    for row in p:
        output.append(np.array(row*features).sum())
    output=output.reshape(-1,1)    
    return output      
