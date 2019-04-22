import numpy as np
import sys
import pandas as pd
import os.path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import defaultdict


from NeuralArch.NeuralNet import *

testFilePath = r"C:\Users\kumar\OneDrive\Documents\Projects\Neuron\DataSet\car.data"


testDF = pd.read_csv(testFilePath)



d = defaultdict(preprocessing.LabelEncoder)

encodedDF = testDF#.sample(n=200)

# Encoding the variable
fit = encodedDF.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
#fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
encodedDF = encodedDF.apply(lambda x: d[x.name].transform(x))



min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(encodedDF.iloc[:,0:6])
df_normalized = pd.DataFrame(np_scaled)




#normalizedDF = pd.concat([df_normalized,testDF.iloc[:,7]],axis=1)

#le = preprocessing.LabelEncoder()
#le.fit(testDF.iloc[:,7])

X_train, X_test, y_train, y_test = train_test_split(
    df_normalized, encodedDF.iloc[:,6], random_state=42)


X_train.loc[:,6] = y_train
X_test.loc[:,6] = y_test


net = NeuralNet(6,5,1,4)

model = net.trainModel(X_train.values,0.3,100)

pred = net.testModel(X_test.values, model)


accuracy = net.calculateAccuracy(X_test.iloc[:,6].values,pred)


print("vv")