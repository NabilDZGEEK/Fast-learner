# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:20:10 2020

@author: nabil
"""
import numpy as np
######## SPLITING
def split(d,l,p):
    s=int(len(d)*(1-p))
    d=np.array(d)
    l=np.array(l).reshape(-1)
    xtrain=d[:s,:]
    xtest=d[s+1:,:]
    ytrain=l[:s]
    ytest=l[s+1:]
    return xtrain,ytrain,xtest,ytest
#########  NORMALIZATION    
def normalize(data):
    data=np.array(data)
    data=data.astype(float)
    for i in range(1):
        mi=data[:,i].min()*1.0
        ma=data[:,i].max()*1.0
        data[:,i]=data[:,i]-mi
        data[:,i]=data[:,i]/(ma-mi)
    return(data)
########## LABEL ENCODER
classes=[]
def fit(x):
    unique=[]
    global classes
    classes=[]
    j=0
    for i in x:
        if not (i in unique) :
            unique.append(i)
            classes.append((i,j))
            j=j+1
    return classes 

def transform(x):
        def get(s):
            for i in classes:
                if i[0]==s :
                    return i[1]
        
        return [get(i) for i in x] 

def fit_transform(x):
    fit(x)
    return transform(x)         