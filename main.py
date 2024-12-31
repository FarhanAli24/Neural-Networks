import numpy as np
import SpiralData
import Layer
import activation
from LossCategoricalCrossentropy import CategoricalCrossentropy

# Create dataset
X, y = SpiralData.generate_spiral_data()

dense1 = Layer.Neural_Layers(2,3)
activ1 = activation.ReLU()

dense2 = Layer.Neural_Layers(3,3)
activ2 = activation.Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

#takes the output of the first layer here
activ1.forward(dense1.output)


#takes the output of activation of first layer as inputs
activ2.forward(activ1.output)

# Make a forward pass of our training. data through this layer
dense2.forward(activ1.output)

#takes the output of second dense layer here
activ2.forward(dense2.output)

#Loss
loss = CategoricalCrossentropy()
print("Loss:", loss.calculate(activ2.output,y))
print(activ2.output[:5])