import numpy as np

class ReLU:
    #Forward pass
    def forward(self, inputs):
        #Calculate output values from inputs
        self.output = np.maximum(0,inputs)

#Softmax Activation
class Softmax:
    #Forward pass
    def forward(self, inputs):
        #Get unnormalized probabilities
        values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))

        #Normalize samples
        probabilities = values/np.sum(values,axis=1,keepdims=True)

        self.output = probabilities
