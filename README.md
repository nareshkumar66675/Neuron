# Neuron

It is a Neural Network Architecture which uses **Forward Backward Propagation**


# Overview

  - Given a Data Set, the architecture finds the best *learning rate, epoch values and number of hidden layers needed*
  - It uses Forward Backward Propagation.
  - The activation function used is sigmoid.
  - Based on the selected parameters, the data is classified and the prediction rate is further analyzed.

# Dataset Used
- Car Evaluation : https://archive.ics.uci.edu/ml/datasets/car+evaluation

- Mushroom Data set : https://archive.ics.uci.edu/ml/datasets/mushroom


# Neural Structure
This is the neural structure for **Car Dataset** with 6 inputs, 6 neurons with 1 hidden layer and 4 output. The Prediction Rate was **92.13%**

![](https://raw.githubusercontent.com/nareshkumar66675/Neuron/master/Charts/NNViz.png)

# Data Set Preprocessing

- All the *numerical attributes* were scaled using min max scaler, to have the same range *0-1*
- All the categorical values were label encoded and then scaled.

# Installation
```
1. Clone the Repository or Download the Project
2. Navigate to the folder
3. Execute 'python Neuron.py'
```


# Sample Execution

#### 1. Select DataSet
```
Select DataSet
1. Car DataSet
2. Mushroom Dataset
Select one Dataset from above : 1
```
#### 2. Find Best Optimal Learning Rate
```
Finding Best Optimal Learning Rate
For Learning Rate : 0.1 the Prediction Rate is 87.27%
For Learning Rate : 0.2 the Prediction Rate is 89.81%
For Learning Rate : 0.3 the Prediction Rate is 91.90%
For Learning Rate : 0.4 the Prediction Rate is 91.67%
For Learning Rate : 0.5 the Prediction Rate is 91.67%
For Learning Rate : 0.6 the Prediction Rate is 90.28%
For Learning Rate : 0.7 the Prediction Rate is 90.05%
For Learning Rate : 0.8 the Prediction Rate is 90.51%
For Learning Rate : 0.9 the Prediction Rate is 90.51%
Optimal Learning Rate '0.3'
```
![CarLearningRate](https://raw.githubusercontent.com/nareshkumar66675/Neuron/master/Charts/CarLearningRate.png "CarLearningRate") 
#### 3. Find the Best Epoch Values
```
Finding Best Epoch Value
For Epoch : 3 the Prediction Rate is 82.64%
For Epoch : 4 the Prediction Rate is 83.33%
For Epoch : 5 the Prediction Rate is 83.56%
For Epoch : 6 the Prediction Rate is 84.49%
For Epoch : 7 the Prediction Rate is 84.95%
For Epoch : 8 the Prediction Rate is 86.34%
For Epoch : 9 the Prediction Rate is 86.57%
For Epoch : 10 the Prediction Rate is 87.50%
For Epoch : 11 the Prediction Rate is 87.96%
For Epoch : 12 the Prediction Rate is 88.66%
For Epoch : 13 the Prediction Rate is 88.66%
For Epoch : 14 the Prediction Rate is 88.66%
For Epoch : 15 the Prediction Rate is 88.89%
For Epoch : 16 the Prediction Rate is 89.35%
For Epoch : 17 the Prediction Rate is 90.74%
For Epoch : 18 the Prediction Rate is 92.13%
For Epoch : 19 the Prediction Rate is 91.90%
Optimal Epoch Value '18'
```
![EpochRate](https://raw.githubusercontent.com/nareshkumar66675/Neuron/master/Charts/CarEpoch.png "EpochRate") 

#### 2. Find Best Optimal Hidden Layers
```
Finding Optimal no of Hidden layers
For Optimal Layer Count : 1 the Prediction Rate is 92.13%
For Optimal Layer Count : 2 the Prediction Rate is 87.50%
For Optimal Layer Count : 3 the Prediction Rate is 68.06%
For Optimal Layer Count : 4 the Prediction Rate is 68.06%
For Optimal Layer Count : 5 the Prediction Rate is 68.06%
Optimal Hidden Layer Count '1'
```
![CarHiddenLayer](https://raw.githubusercontent.com/nareshkumar66675/Neuron/master/Charts/CarHiddenLayer.png "CarHiddenLayer") 


# Project Struture
##### ExpMaxML
- **Neuron.py** - Main Startup File.
- **/NeuralArch**
    - NeuralNet - Class Implementation of Neural Net architecture
##### Charts
- Various Charts
##### DataSet
- car.data
- agaricus-lepiota.data


  
