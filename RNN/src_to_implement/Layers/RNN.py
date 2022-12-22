import numpy as np
import warnings
import copy

from Layers.Base import BaseLayer

class RNN(BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass