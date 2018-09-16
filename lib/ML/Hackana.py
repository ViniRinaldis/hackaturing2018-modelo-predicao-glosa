# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 07:40:40 2018

@author: Marcelo
"""

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import pickle as pk


# Read the data set
data = pd.read_csv('filthack.csv', encoding='utf-8', error_bad_lines=False,sep=';')

# Creating numerical fields normal statistically distributed
data.head()
Xobj = data.iloc[:, 0:4]
Xreal = data.iloc[:, 4:7]
Y = data.iloc[:, 7]

# Solving the objects problem -- Characteristic fields
le = preprocessing.LabelEncoder()


size = len(Y)
X = np.zeros([size,7])
for i in range(4):
    le.fit(Xobj.iloc[:,i])
    X[:,i] = le.transform(Xobj.iloc[:,i])
    pk.dump(le,open('le'+str(i)+'.pkl','wb'))
X[:,4:7] = Xreal

## Segregating the negative data from positive
Y1, X1 = np.zeros([len(Y),1]), np.zeros([len(Y),7])
Y0, X0 = np.zeros([len(Y),1]), np.zeros([len(Y),7])
s1, s2 = 0, 0

# Creating numerical fields statistically distributed
for i in range(len(Y)):
    if Y.iloc[i] == 1:
        Y1[s1] = Y.iloc[i]
        X1[s1,:] = X[i,:]
        s1 = s1 + 1
    else:
        Y0[s2] = Y.iloc[i]
        X0[s2,:] = X[i,:]
        s2 = s2 + 1

Y0 = Y0[0:s2]
X0 = X0[0:s2]
Y1 = Y1[0:s1]
X1 = X1[0:s1]

#Ensure the same proportion as positives and negatives categories
total = len(Y0)
partial = len(Y1)
prop = partial/total

## Shuffling the data
X_use, X_discart, Y_use, Y_discart = train_test_split( X0, Y0, \
                                                    test_size = 0.3, \
                                                    train_size = prop, \
                                                    random_state = 90)

## Reconstruct same proportion data--set
X_data = np.concatenate([X1,X_use])
Y_data = np.concatenate([Y1,Y_use])


## Normalization
for i in range(7):
    x_max = max(X_data[:,i])
    x_min = min(X_data[:,i])
    k = float(1/(x_max - x_min))
    X_data[:,i] = (X_data[:,i] - x_min)*k  

# Data prep to trainning such as test 
prop = 12000/len(Y_data)
X_train, X_test, y_train, y_test = train_test_split( X_data, Y_data, \
                                                    test_size = 0.3, \
                                                    train_size = 0.7, \
                                                    random_state = 100)


# Estimating the tree model - such as the ensemble comparison
RFC_Model = RandomForestClassifier(max_depth=20, random_state=100,\
                                   n_estimators=15)
RFC_Model.fit(X_train,y_train.ravel())
# print(RFC_Model.feature_importances_)

# Creating the model for automatic web api
pk.dump(RFC_Model,open('model.pkl','wb'))

# Predicting the output data
y_RFC_pred = RFC_Model.predict(X_test)

# Performance measure
accuracy_score(y_test,y_RFC_pred)*100

# Another one
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_RFC_pred)
acc = 100*(cm[0,0] + cm[1,1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])

print(acc)

# Predicting all data classification
y_RFC_pred_train = RFC_Model.predict(X_train)
accuracy_score(y_train,y_RFC_pred_train)*100

