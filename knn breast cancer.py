# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:58:48 2018

@author: Akshay Kumar
"""

import numpy as np
from sklearn import cross_validation, neighbors 
import pandas as pd

data = pd.read_csv('cancer1.csv')
print(data)
#print(df)
# Replacing '?' with -99999 as NaN Value
 
data.replace('?',-99999, inplace=True)
#dropping the first column referred as 'id'
#1 IS FOR THE column no
data.drop(['id'], 1, inplace=True)

#X is a features y is a label 'class'
#x is the data where all the columns execpt class is saved
X = np.array(data.drop(['class'], 1))
y = np.array(data['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# n_neighbors = 5 (default)  
#we here gave value of k as 17
knn = neighbors.KNeighborsClassifier(n_neighbors=17)

# we're using the K Nearest Neighbors classifier from Sklearn
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print("Accuracy Result")
print(accuracy)

## ------------------------------------------------------------
#Prediction part
#Take an example values to predict on which part it will fall
new_data = np.array([4,2,1,5,9,2,6,2,1])
print (new_data)
#Reshaping the values
new_data = new_data.reshape(1, -1)
#Prediction
prediction = knn.predict(new_data)
print("Prediction Class")
print (prediction)
new_data = np.array([[4,9,1,9,1,2,6,8,1],[4,2,1,1,1,2,3,9,9]])
new_data = new_data.reshape(2, -1)
print(new_data)
prediction = knn.predict(new_data)
print(prediction)
new_data = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
print(new_data)
new_data = new_data.reshape(len(new_data), -1)
prediction = knn.predict(new_data)
print(prediction)