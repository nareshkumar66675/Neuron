import numpy as np
import sys
import pandas as pd
import os.path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import defaultdict
from NeuralArch.NeuralNet import *
import seaborn as sns;
sns.set()

import matplotlib.pylab as plt





def GetMushroomData():
    testFilePath = r"C:\Users\kumar\OneDrive\Documents\Projects\Neuron\DataSet\agaricus-lepiota.data"
    rawDF = pd.read_csv(testFilePath,header=None)

    # Reorder Decision Attribute for consistency
    mushroomDF = rawDF.iloc[:,1:]
    mushroomDF.loc[:,len(rawDF.columns)] = rawDF.iloc[:,0]

    return mushroomDF

def GetCarData():
    testFilePath = r"C:\Users\kumar\OneDrive\Documents\Projects\Neuron\DataSet\car.data"
    rawDF = pd.read_csv(testFilePath)

    ## Reorder Decision Attribute for consistency
    #mushroomDF = rawDF.iloc[:,1:]
    #mushroomDF.loc[:,len(encodedDF.columns)] = rawDF.iloc[:,0]

    return rawDF

def GetLearningRate(selectedDF,X_train,X_test,y_test ):
    # return 0.3
    lRateAccuracy = {}
    for lRate in range(1,10):
            net = NeuralNet(len(selectedDF.columns)-1,len(selectedDF.columns)-1,1,len(selectedDF.iloc[:,-1].unique()))

            model = net.trainModel(X_train.values,lRate/10,20)

            pred = net.testModel(X_test.values, model)

            accuracy = net.calculateAccuracy(y_test.values,pred)

            print("For Learning Rate : {0} the Prediction Rate is {1}%".format(lRate/10,"{0:.2f}".format(accuracy)))

            lRateAccuracy[lRate/10] = accuracy

    lists = sorted(lRateAccuracy.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    #plt.plot(x, y)

    sns.lineplot(x=x, y=y)
    plt.show()

    return max(lRateAccuracy, key=lRateAccuracy.get)

def GetOptimalEpoch(selectedDF,X_train,X_test,y_test,lRate ):

    #return 9
    epochAccuracy = {}
    for epoch in range(3,20):
            net = NeuralNet(len(selectedDF.columns)-1,len(selectedDF.columns)-1,1,len(selectedDF.iloc[:,-1].unique()))

            model = net.trainModel(X_train.values,lRate,epoch)

            pred = net.testModel(X_test.values, model)

            accuracy = net.calculateAccuracy(y_test.values,pred)

            print("For Epoch : {0} the Prediction Rate is {1}%".format(epoch,"{0:.2f}".format(accuracy)))

            epochAccuracy[epoch] = accuracy

    lists = sorted(epochAccuracy.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    #plt.plot(x, y)

    sns.lineplot(x=x, y=y)
    plt.show()

    return max(epochAccuracy, key=epochAccuracy.get)

def GetOptimalHiddenLayers(selectedDF,X_train,X_test,y_test,lRate,epoch ):
    hiddenLayerAccuracy = {}
    for hiddenLayer in range(1,6):
            net = NeuralNet(len(selectedDF.columns)-1,len(selectedDF.columns)-1,hiddenLayer,len(selectedDF.iloc[:,-1].unique()))

            model = net.trainModel(X_train.values,lRate,epoch)

            pred = net.testModel(X_test.values, model)

            accuracy = net.calculateAccuracy(y_test.values,pred)

            print("For Optimal Layer Count : {0} the Prediction Rate is {1}%".format(hiddenLayer,"{0:.2f}".format(accuracy)))

            hiddenLayerAccuracy[hiddenLayer] = accuracy

    lists = sorted(hiddenLayerAccuracy.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    #plt.plot(x, y)

    sns.lineplot(x=x, y=y)
    plt.show()

    return max(hiddenLayerAccuracy, key=hiddenLayerAccuracy.get)

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


    #net = NeuralNet(len(selectedDF.columns)-1,10,1,len(selectedDF.iloc[:,-1].unique()))

    #model = net.trainModel(X_train.values,0.3,20)

    #pred = net.testModel(X_test.values, model)

    #accuracy = net.calculateAccuracy(y_test.values,pred)

    #print("Prediction Rate '{0}'".format("{0:.2f}".format(accuracy)))

    #print("Prediction Rate" + accuracy)

    #lRateAccuracy = {}
    #for lRate in range(1,10):
    #        net = NeuralNet(len(selectedDF.columns)-1,10,1,len(selectedDF.iloc[:,-1].unique()))

    #        model = net.trainModel(X_train.values,lRate/10,20)

    #        pred = net.testModel(X_test.values, model)

    #        accuracy = net.calculateAccuracy(y_test.values,pred)

    #        lRateAccuracy[lRate/10] = accuracy

    #lists = sorted(lRateAccuracy.items()) # sorted by key, return a list of tuples

    #x, y = zip(*lists) # unpack a list of pairs into two tuples

    ##plt.plot(x, y)

    #sns.lineplot(x=x, y=y)
    #plt.show()

    #for i in lRateAccuracy:
    #    print(i, lRateAccuracy[i])
    print("Finding Best Optimal Learning Rate")
    optimalLearningRate = GetLearningRate(selectedDF,X_train, X_test,y_test )
    print("Optimal Learning Rate '{0}'".format(optimalLearningRate))


    print("Finding Best Epoch Value")
    epochValue = GetOptimalEpoch(selectedDF,X_train, X_test,y_test,optimalLearningRate )
    print("Optimal Epoch Value '{0}'".format(epochValue))

    print("Finding Optimal no of Hidden layers")
    hiddenLayerCount = GetOptimalHiddenLayers(selectedDF,X_train, X_test,y_test,optimalLearningRate,epochValue )
    print("Optimal Hidden Layer Count '{0}'".format(hiddenLayerCount))
    


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