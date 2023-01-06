import numpy as np

from Layers import TanH, Sigmoid, FullyConnected
from Layers.Base import BaseLayer  # BaseLayer is the base class for all layers
import warnings
import copy


class RNN(BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.time_step = None
        self._weights = None
        self.forward_output = np.array([])
        self._gradient_weights = None
        self._memorize = False

        # Initialize fully connected layers
        self.h_layer = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)

        self.y_layer = FullyConnected.FullyConnected(self.hidden_size, self.output_size)

        self.tanH = TanH.TanH()
        self.sigmoid = Sigmoid.Sigmoid()
        
        self.trainable = True

        self.batch_size = None

        self.input_tensor = None
        self.hidden_state = np.zeros((hidden_size, 1))

        self.initialize()

        # self.weights = np.random.rand(input_size + hidden_size + 1, hidden_size)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize_value):
        self._memorize = memorize_value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.time_step = input_tensor.shape[0]

        self.forward_output = np.zeros((self.time_step, self.output_size))

        for i in range(self.time_step):
            # concatenate the input tensor with the hidden state
            x_new = np.concatenate((input_tensor[i].reshape(input_tensor[i].size, 1), self.hidden_state)).T
            # Calculate the output of h layer
            h_value = self.h_layer.forward(x_new)
            self.hidden_state = self.tanH.forward(h_value.T)
            # Calculate the output of y layer
            y_value = self.y_layer.forward(self.hidden_state.T)
            self.forward_output[i, :] = self.sigmoid.forward(y_value)

        return self.forward_output

    def backward(self, error_tensor):

        self.error_tensor = error_tensor

        for i in range(self.time_step-1, -1, -1):


        return self.error_tensor



    def initialize(self, weights_initializer=None, bias_initializer=None):

        # Initialize the weights of the h layer
        self.h_layer.initialize(weights_initializer, bias_initializer)
        # Initialize the weights of the y layer
        self.y_layer.initialize(weights_initializer, bias_initializer)

        self.weights = self.h_layer.weights
        return self.weights

