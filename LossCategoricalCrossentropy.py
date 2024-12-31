import numpy as np

class Loss:
    def calculate(self,output,y):
        #Calculate sample losses
        sample_losses = self.forward(output,y)

        #Calculate Mean losses
        loss = np.mean(sample_losses)

        return loss

#Crossentropy
class CategoricalCrossentropy(Loss):
    def forward(self, predicted, trueVales):
        samples = len(predicted)

        #Prevent divide by 0
        predictedClipped = np.clip(predicted, 1e-7, 1 - 1e-7)


        #Probabilities for target values - only if categorical labels
        if len(trueVales.shape) == 1:
            confidence = predictedClipped[range(samples),trueVales]

        #Mask Values - only for one-hot encoded labels
        elif len(trueVales.shape) == 2:
            confidence = np.sum(predictedClipped * trueVales, axis=1)

        #Losses
        negative_loss_likelihood = -np.log(confidence)
        return negative_loss_likelihood