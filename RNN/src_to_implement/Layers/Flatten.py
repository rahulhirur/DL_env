import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    
    def __init__(self):
        
        super().__init__()
        self.fwd_shape = None
        self.forward_output = None
        self.backward_output = None

        
    def forward(self, input_tensor):

        self.fwd_shape = input_tensor.shape
        self.forward_output = input_tensor.reshape(input_tensor.shape[0], int(input_tensor.size/input_tensor.shape[0]))
        return self.forward_output

    
    def backward(self, error_tensor):

        self.backward_output = error_tensor.reshape(self.fwd_shape)
        return self.backward_output
