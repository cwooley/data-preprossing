# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:37:21 2017

@author: Charles
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


np.set_printoptions(threshold=np.nan)

# Importing the dataset
dataset = pd.read_csv('Data.csv')

#Creating matrix of features(independant variables)
X = dataset.iloc[:, :-1].values

#Creating Vector of dependant variables
Y = dataset.iloc[:, 3].values
                
#Dealing with missing data
#Import Imputer from sci kit learn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


#We need to encode categorical variables into numbers... things like countries, names ect.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
#Dummy Encoding

# Dummy encoding is necessary to make sure that your ML Algos don't think one category
# is greater than another
# This is accomplished by seperating into multiple columns with boolean values instead.
# This can be accomplished with oneHotEncoder

#Splitting the dataset into the Training set and Test set
#Training set is what the model will learn on, and the Test set is what we use 
#to test our models preformance.
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Feature Scaling
#Feature Scaling is necessary because we need to make sure that one value does not get 
#dominated by another due to different scales.
#this is done by standardization or normalization

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Do we need to scale the dummy variables?
# Some say no, some say yes.
# It really depends on context.














