# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 17:19:37 2018

@author: Marcelo
"""
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from sklearn import tree
from sklearn import preprocessing
import pandas as pd

# Read the data set
data = pd.read_csv('filthack.csv', encoding='utf-8', error_bad_lines=False,sep=';')

# Segregating the data tyes for pre--treatment
data.head()
Xobj = data.iloc[:, 0:4]
Xreal = data.iloc[:, 4:7]
Y = data.iloc[:, 7]

# Solving the objects problem -- Characteristic fields
le = preprocessing.LabelEncoder()

# Creating numerical fields normal statistically distributed
size = len(Y)
X = np.zeros([size,7])
for i in range(4):
    le.fit(Xobj.iloc[:,i])
    X[:,i] = le.transform(Xobj.iloc[:,i])
X[:,4:7] = Xreal

## Segregating the negative data from positive
Y1, X1 = np.zeros([len(Y),1]), np.zeros([len(Y),7])
Y0, X0 = np.zeros([len(Y),1]), np.zeros([len(Y),7])
s1, s2 = 0, 0

# Getting the separete output classes
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

# Ensuring the same proportion of positives and negatives
total = len(Y0)
partial = len(Y1)
prop = partial/total

## Shuffling the data set
X_use, X_discart, Y_use, Y_discart = train_test_split( X0, Y0, \
                                                    test_size = 0.3, \
                                                    train_size = prop, \
                                                    random_state = 90)

## Reconstruct same proportion data--set
X_data = np.concatenate([X1,X_use])
Y_data = np.concatenate([Y1,Y_use])


############# Decision tree classifier Training ############################

#### Prepare data objects
X_train, X_test, y_train, y_test = train_test_split( X_data, Y_data, \
                                                    test_size = 0.3,\
                                                    train_size = 0.7, \
                                                    random_state = 100)


# Decision tree model estimation
Gini_Model = DecisionTreeClassifier(criterion = "gini", random_state = 100,\
                               max_depth=4, min_samples_leaf=12)
Gini_Model.fit(X_train, y_train)

# Predicting the tree output
y_pred = Gini_Model.predict(X_test)

# Considering only the glosna datas to consider possible mis--performance
l = 0
y_comp = np.zeros([len(y_pred),1])
y_sol = np.zeros([len(y_pred),1])
for k in range(len(y_pred)):
    f = y_test[k]
    if f == 1:
        y_comp[l] = y_test[k]
        y_sol[l] = y_pred[k]
        l = l + 1
y_comp_1 = y_comp[0:l]
y_sol_1 = y_sol[0:l]

#accuracy_score(y_test,y_pred)*100

# Performance measure
accuracy_score(y_comp_1,y_sol_1)*100

#################### Cross--Validation ########################################
from sklearn.model_selection import cross_val_score

scores_dt = cross_val_score(Gini_Model, X_train, y_train,
                            scoring='accuracy', cv=5)
print(scores_dt.mean()*100)

######################## SVM algorithm #####################################
from sklearn import svm

## Normalization
for i in range(7):
    x_max = max(X_data[:,i])
    x_min = min(X_data[:,i])
    k = float(1/(x_max - x_min))
    X_data[:,i] = (X_data[:,i] - x_min)*k  

# Data prep
prop = 8000/len(Y_data)
X_train, X_test, y_train, y_test = train_test_split( X_data, Y_data, \
                                                    test_size = 0.3, \
                                                    train_size = prop, \
                                                    random_state = 100)

# Determines lagrange multipliers as the parameter vectors
SVM_Model = svm.SVC()
SVM_Model.fit(X_train, y_train.ravel())

# Predict the output data
y_svm_pred = SVM_Model.predict(X_test)

# Considering only the glosa datas to consider possible mis--performance
l = 0
y_svm_comp = np.zeros([len(y_svm_pred),1])
y_svm_sol = np.zeros([len(y_svm_pred),1])
for k in range(len(y_svm_pred)):
    f = y_test[k]
    if f == 1:
        y_svm_sol[l] = y_svm_pred[k]
        y_svm_comp[l] = y_test[k]
        l = l + 1
y_svm_comp_1 = y_svm_comp[0:l]
y_svm_sol_1 = y_svm_sol[0:l]

#accuracy_score(y_svm_comp_1,y_svm_sol_1)*100

# Performance measure
accuracy_score(y_test,y_svm_pred)*100

# Another performance measure
from sklearn.model_selection import cross_val_score
scores_dt = cross_val_score(SVM_Model, X_train, y_train.ravel(),
                            scoring='accuracy', cv=5)
print(scores_dt.mean()*100)

############################ Naive Bayes #####################################
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

# Data prep
prop = 12000/len(Y_data)
X_train, X_test, y_train, y_test = train_test_split( X_data, Y_data, \
                                                    test_size = 0.3, \
                                                    train_size = prop, \
                                                    random_state = 100)

#Model estimation
#Naive_Model = GaussianNB()
#Naive_Model = MultinomialNB()
Naive_Model = BernoulliNB()
Naive_Model.fit(X_train, y_train)

# Predicting the output of the classifier
y_naive_pred = Naive_Model.predict(X_test)

# Considering only the glosa datas to consider possible mis--performance
l = 0
y_naive_comp = np.zeros([len(y_naive_pred),1])
y_naive_sol = np.zeros([len(y_naive_pred),1])
for k in range(len(y_naive_pred)):
    f = y_test[k]
    if f == 1:
        y_naive_sol[l] = y_naive_pred[k]
        y_naive_comp[l] = y_test[k]
        l = l + 1
y_naive_comp_1 = y_naive_comp[0:l]
y_naive_sol_1 = y_naive_sol[0:l]

#accuracy_score(y_naive_comp_1,y_naive_sol_1)*100

# Performance measure 
accuracy_score(y_test,y_naive_pred)*100


################# Neural Network ###########################################
from sklearn.neural_network import MLPClassifier

## Normalization
for i in range(7):
    x_max = max(X_data[:,i])
    x_min = min(X_data[:,i])
    k = float(1/(x_max - x_min))
    X_data[:,i] = (X_data[:,i] - x_min)*k  

# Data prep, using 1000 points as example
prop = 1000/len(Y_data)
X_train, X_test, y_train, y_test = train_test_split( X_data, Y_data, \
                                                    test_size = 0.3, \
                                                    train_size = prop, \
                                                    random_state = 100)


# Creating the Neural Network Classifier
MLP_Model = MLPClassifier(solver='lbfgs', alpha=1e-5, \
                    hidden_layer_sizes=(5, 4), random_state=1)
MLP_Model.fit(X_train, y_train.ravel())

# Predicting the output
y_MLP_pred = MLP_Model.predict(X_test)


# Considering only the glosa datas to consider possible mis--performance
l = 0
y_MLP_comp = np.zeros([len(y_MLP_pred),1])
y_MLP_sol = np.zeros([len(y_MLP_pred),1])
for k in range(len(y_MLP_pred)):
    f = y_test[k]
    if f == 1:
        y_MLP_sol[l] = y_MLP_pred[k]
        y_MLP_comp[l] = y_test[k]
        l = l + 1
y_MLP_comp_1 = y_MLP_comp[0:l]
y_MLP_sol_1 = y_MLP_sol[0:l]

#accuracy_score(y_naive_comp_1,y_naive_sol_1)*100

# Performance measure
accuracy_score(y_test,y_MLP_pred)*100




######### Deeppp Learning ##########################################
import keras
from keras.models import Sequential
from keras.layers import Dense

## Normalization
for i in range(7):
    x_max = max(X_data[:,i])
    x_min = min(X_data[:,i])
    k = float(1/(x_max - x_min))
    X_data[:,i] = (X_data[:,i] - x_min)*k  

# Data prep
prop = 4000/len(Y_data)
X_train, X_test, y_train, y_test = train_test_split( X_data, Y_data, \
                                                    test_size = 0.3, \
                                                    train_size = prop, \
                                                    random_state = 100)


classifier = Sequential()

######### Initializing layers

classifier.add(Dense(output_dim = 7, \
                     init = 'uniform', activation = 'relu', input_dim = 7))

classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', \
                   loss = 'binary_crossentropy', metrics = ['accuracy'])

######## Final layers


# Model estimation
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Model performance
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
acc = 100*(cm[0,0] + cm[1,1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])

print(acc)


###################### Random Forest Predict ################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

## Normalization
for i in range(7):
    x_max = max(X_data[:,i])
    x_min = min(X_data[:,i])
    k = float(1/(x_max - x_min))
    X_data[:,i] = (X_data[:,i] - x_min)*k  

# Data prep
prop = 12000/len(Y_data)
X_train, X_test, y_train, y_test = train_test_split( X_data, Y_data, \
                                                    test_size = 0.3, \
                                                    train_size = prop, \
                                                    random_state = 100)

# Model estimation
RFC_Model = RandomForestClassifier(max_depth=20, random_state=100,\
                                   n_estimators=15)
RFC_Model.fit(X_train,y_train.ravel())
# print(RFC_Model.feature_importances_)

# Prediction
y_RFC_pred = RFC_Model.predict(X_test)

# One performance measure
accuracy_score(y_test,y_RFC_pred)*100

# Second performance measure
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_RFC_pred)
acc = 100*(cm[0,0] + cm[1,1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])

print(acc)

# Predict all data
y_RFC_pred_train = RFC_Model.predict(X_train)
accuracy_score(y_train,y_RFC_pred_train)*100





