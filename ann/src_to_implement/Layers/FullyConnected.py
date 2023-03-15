import numpy as np

# FullyConnected layer inherits the base layer
from Layers.Base import BaseLayer

from Optimization import Optimizers


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):

        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True

        self.input_tensor = None
        self.error_tensor = None
        
        self._optimizer = None
        self._gradient_weights = None
        
        """"
        - Initialize weights uniformly random in the range [0, 1) 
        - Including bias in last row of weight matrix allows:
          single matrix multiplication
        """
        self.weights = np.random.rand(self.input_size + 1, 
                                      self.output_size)
        
        self.forward_output = None
        self.backward_output = None

        
    def forward(self, input_tensor):
        
        """"
        - Returns a tensor that serves as the input_tensor to the next layer
        
        - input_tensor: rows = batch_size, columns = input_size
        - batch_size: number of inputs processed simultaneously
        - output_size: number of columns of the output
        
        - A column of ones is connected to the input_tensor in order to:
          multiply it with the last row of biases in the weight matrix
        """"
        
        self.input_tensor = np.concatenate((input_tensor, 
                                            np.ones((input_tensor.shape[0], 1))), axis=1)
        self.forward_output = np.dot(self.input_tensor, self.weights)
        
        return self.forward_output

    
    """"
    setter & getter property optimizer: 
        sets and returns the protected member _optimizer for this layer
    """"
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_value):
        self._optimizer = optimizer_value

    """"
    - For future reasons, property gradient_weights is provided
    - It returns the gradient with respect to the weights, 
      after they have been calculated in the backward-pass
    """"
    
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

        
    def backward(self, error_tensor):
        
        """"
        - Returns a tensor that serves as the error_tensor for the previous layer
        - Reminder! In the backward pass, we are going in a direction opposite to
          the forward pass
        """"

        self.backward_output = np.dot(error_tensor, self.weights.transpose())
        self.gradient_weights = np.dot(self.input_tensor.transpose(), error_tensor)
        
        # Don’t perform an update if the optimizer is not set
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return self.backward_output[:, :-1] # ':' selects all rows, ':-1' all but the last column
