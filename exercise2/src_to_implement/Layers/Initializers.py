import numpy as np

class Constant:
    def __init__(self, constant_value = 0.1):
        self.constant_value = constant_value
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.random.rand(self.weights_shape, self.fan_in, self.fan_in)
