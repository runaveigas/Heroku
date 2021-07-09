# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:15:34 2021

@author: RUNA
"""


import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


df = pd.read_excel('bloodpress.xlsx')
#print(df.isnull().values.sum())

X = df.iloc[:,1:]
y = df.iloc[:,0:1]

regressor = LinearRegression()
regressor.fit(X,y)

# Make predictions using the testing set
bp_predict = regressor.predict(X)

# Saving model to disk
pickle.dump(regressor,open('model.pkl','wb'))

    