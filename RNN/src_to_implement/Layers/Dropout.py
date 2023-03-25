import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):

    def __init__(self, probability):
        
        super().__init__()
        self.mask = None
        self.probability = probability
        self.forward_output = None
        self.backward_output = None

        
    def forward(self, input_tensor):
        
        # create a mask of the same shape as the input_tensor
        self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape)
        
        # apply the mask to the input_tensor
        if self.testing_phase:
            self.forward_output = input_tensor * 1
        else:
            self.mask = self.mask * (1 / self.probability)
            self.forward_output = input_tensor * self.mask

        return self.forward_output

    
    def backward(self, error_tensor):
        
        # apply the mask to the error_tensor
        self.backward_output = error_tensor * self.mask
        
        return self.backward_output
