# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 07:46:59 2018

@author: Marcelo
"""
import numpy as np
#from sklearn.cross_validation import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn import preprocessing
#import pandas as pd
import pickle as pk
import sys

# Receive the data from API, as the example above
data = sys.argv[1].split(";")
#data = ['ELETIVO','43979','F','MEDICAMENTOS',62.271,2,98]  #1

# Opening the .pkl models pre fitted 
with open("model.pkl","rb") as f:
    model = pk.load(f)

with open("le0.pkl","rb") as l0:
    le0 = pk.load(l0)

with open("le1.pkl","rb") as l1:
    le1 = pk.load(l1)
    
with open("le2.pkl","rb") as l2:
    le2 = pk.load(l2)
    
with open("le3.pkl","rb") as l3:
    le3 = pk.load(l3)

X = np.zeros([1,7])

# Determining the Input variables for the model
X[0,1] = le0.transform([data[0]])
X[0,1] = le1.transform([data[1]])
X[0,2] = le2.transform([data[2]])
X[0,3] = le3.transform([data[3]])
X[0,4] = data[4]
X[0,5] = data[5]
X[0,6] = data[6]

# Predicting the output model
prediction = model.predict(X)

# Sending the output ot the API 
print(int(prediction))
