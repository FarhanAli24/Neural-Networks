import numpy as np

class Neural_Layers:
    #Layer Initialization
    def __init__(self, inputs, neurons):
        #Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))

    #Forward pass
    def forward(self,inputs):
        #Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
