from random import seed
from random import randrange
from random import random
from math import exp


class NeuralNet(object):
    """Neural Network"""

    def __init__(self, inputCount, hiddenNeuronCount, hiddenLayerCount, outputCount):
        self.inptCount = inputCount
        self.hiddenNeuronCount = hiddenNeuronCount
        self.hiddenLayerCount = hiddenLayerCount
        self.outputCount = outputCount
        self.layers = list()

        randomSeed = 30

        #Construct Hidden Layer
        for x in range(0,hiddenLayerCount):
            seed(randomSeed)
            randomSeed+hiddenLayerCount
            hiddenLayer = [{'values':[random() for i in range(inputCount + 1)]} for i in range(hiddenNeuronCount)]
            self.layers.append(hiddenLayer)

        #Output layer
        seed(1)
        outputLayer = [{'values':[random() for i in range(hiddenNeuronCount + 1)]} for i in range(outputCount)]
        self.layers.append(outputLayer)



    # Sigmoid Function
    def sigmoid(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    # Sigmoid's derivative function
    def sigmoidDerivative(self, output):
        return output * (1.0 - output)


    # Calculate Accuracy
    def calculateAccuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Neuron Activation for the given input
    def activateNeuron(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(inputs)-1):
            activation += weights[i] * inputs[i]
        return activation

    # Updates the weights after the backward propagation
    def updateWeights(self, layers, row, lRate):
        for i in range(len(layers)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in layers[i - 1]]
            for neuron in layers[i]:
                for j in range(len(inputs)):
                    neuron['values'][j] += lRate * neuron['cost'] * inputs[j]
                neuron['values'][-1] += lRate * neuron['cost']

    # Forward Propagation - Apply the Activation and function
    def forwardPropagate(self, layers, row):
        inputs = row
        for layer in layers:
            new_inputs = []
            for neuron in layer:
                activation = self.activateNeuron(neuron['values'], inputs)
                neuron['output'] = self.sigmoid(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Backward Propagation - Calculate loss
    def backwardPropagate(self, layers, expected):
        for i in reversed(range(len(layers))):
            layer = layers[i]
            errors = list()
            if i != len(layers)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in layers[i + 1]:
                        error += (neuron['values'][j] * neuron['cost'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['cost'] = errors[j] * self.sigmoidDerivative(neuron['output'])


    # Predict based on the model and data
    def predict(self, layers, row):
        outputs = self.forwardPropagate(layers, row)
        return outputs.index(max(outputs))

    # Trains a model based on the passed data
    def trainModel(self, train, learnRate, epochCount):
        for epoch in range(epochCount):
            for row in train:
                outputs = self.forwardPropagate(self.layers, row)
                expected = [0 for i in range(self.outputCount)]
                val = row[-1]
                expected[int(val)] = 1
                self.backwardPropagate(self.layers, expected)
                self.updateWeights(self.layers, row, learnRate)

        return self.layers

    # Tests model and returns accuracy based on the model and test data
    def testModel(self, test, model):
        predictions = list()
        for row in test:
            prediction = self.predict(model, row)
            predictions.append(prediction)
        return(predictions)
    


