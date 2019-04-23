import numpy as np
import sys
import pandas as pd
import os.path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import defaultdict
from NeuralArch.NeuralNet import *






def GetMushroomData():
    testFilePath = r"C:\Users\kumar\OneDrive\Documents\Projects\Neuron\DataSet\agaricus-lepiota.data"
    rawDF = pd.read_csv(testFilePath,header=None)

    # Reorder Decision Attribute for consistency
    mushroomDF = rawDF.iloc[:,1:]
    mushroomDF.loc[:,len(encodedDF.columns)] = rawDF.iloc[:,0]

    return mushroomDF

def GetCarData():
    testFilePath = r"C:\Users\kumar\OneDrive\Documents\Projects\Neuron\DataSet\car.data"
    rawDF = pd.read_csv(testFilePath)

    ## Reorder Decision Attribute for consistency
    #mushroomDF = rawDF.iloc[:,1:]
    #mushroomDF.loc[:,len(encodedDF.columns)] = rawDF.iloc[:,0]

    return rawDF

while True:
    print("Select DataSet")
    print("1. Car DataSet")
    print("2. Mushroom Dataset")

    dataChoice = int(input('Select one Dataset from above : '))

    if dataChoice == 1:
        selectedDF = GetCarData()
    elif dataChoice == 2:
        selectedDF = GetMushroomData()
    else:
        choice = input('Enter Valid Option. Press Y to Restart and N to Exit: ')
        if str.lower(choice) == 'n':
            sys.exit()
        else:
            continue


    dfLabelEncoder = defaultdict(preprocessing.LabelEncoder)
    # Encoding the variable
    fit = selectedDF.apply(lambda x: dfLabelEncoder[x.name].fit_transform(x))

    # Inverse the encoded
    #fit.apply(lambda x: d[x.name].inverse_transform(x))

    # Using the dictionary to label future data
    selectedDF = selectedDF.apply(lambda x: dfLabelEncoder[x.name].transform(x))

    
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(selectedDF.iloc[:,0:len(selectedDF.columns)])
    df_normalized = pd.DataFrame(np_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
    df_normalized, selectedDF.iloc[:,-1], random_state=42)


    X_train.loc[:,len(selectedDF.columns)] = y_train
    X_test.loc[:,len(selectedDF.columns)] = y_test


    net = NeuralNet(len(selectedDF.columns)-1,10,1,len(selectedDF.iloc[:,-1].unique()))

    model = net.trainModel(X_train.values,0.3,20)

    pred = net.testModel(X_test.values, model)

    accuracy = net.calculateAccuracy(y_test.values,pred)

    print("Prediction Rate '{0}'".format("{0:.2f}".format(accuracy)))

    print("Prediction Rate" + accuracy)

print("End")




##categoricalColnsDF = testDF[['a1','a4','a5','a6','a8','a9','a11','a12']]
##numericalColnsDF = testDF[['a2','a3','a7','a10','a13','a14']]


### Encoding the variable
##fit = categoricalColnsDF.apply(lambda x: dfLabelEncoder[x.name].fit_transform(x))

### Inverse the encoded
###fit.apply(lambda x: d[x.name].inverse_transform(x))

### Using the dictionary to label future data
###categoricalColnsDF = categoricalColnsDF.apply(lambda x: dfLabelEncoder[x.name].transform(x))



##for x in numericalColnsDF.columns:
##    categoricalColnsDF.loc[:,x] = numericalColnsDF[x]

##categoricalColnsDF.loc[:,"class"] = testDF["class"]