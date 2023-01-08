import numpy as np

from Layers.Base import BaseLayer
from Optimization import Optimizers


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.norm_sum = 0
        self.backward_output = None
        self.input_tensor = None
        self._optimizer = None
        self._gradient_weights = None
        self.forward_output = None
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        # Initialize weights to carry bias in last row
        self.weights = np.random.rand(input_size+1, output_size)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_value):
        self._optimizer = optimizer_value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def forward(self, input_tensor):
        # wx+b*1

        self.input_tensor = np.concatenate((input_tensor, np.ones((input_tensor.shape[0], 1))), axis=1)

        self.forward_output = np.dot(self.input_tensor, self.weights)

        if self.optimizer is not None:
            if self.optimizer.regularizer is not None:
                self.norm_sum = self.optimizer.regularizer.norm(self.weights)

        return self.forward_output

    def backward(self, error_tensor):

        self.backward_output = np.dot(error_tensor, self.weights.transpose())
        self.gradient_weights = np.dot(self.input_tensor.transpose(), error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return self.backward_output[:, :-1]

    # This is extra when compared to previous exercise, The method intakes weights and bias and concatenate the data
    def initialize(self, weights_initializer=None, bias_initializer=None):
        fan_in = self.input_size
        fan_out = self.output_size
        weight_shape = (fan_in, fan_out)
        bias_shape = (1, fan_out)

        if weights_initializer is None:
            weight_shape = (fan_in+1, fan_out)
            self.weights = np.random.rand(*weight_shape)
        else:
            self.weights = np.concatenate((weights_initializer.initialize(weight_shape, fan_in, fan_out), bias_initializer.initialize(bias_shape, fan_in, fan_out)), axis=0)
