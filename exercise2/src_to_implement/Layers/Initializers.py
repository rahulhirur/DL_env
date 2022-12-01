import numpy as np


class Constant:

    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value
        self.weights = None

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        self.weights = np.full(weights_shape, self.constant_value)

        return self.weights


class UniformRandom:

    def __init__(self):
        self.weights = None

    def initialize(self, weights_shape, fan_in=None, fan_out=None):

        self.weights = np.random.rand(weights_shape[0], weights_shape[1])
        return self.weights


class Xavier:

    def __init__(self):
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        # fan_in =  input_channels *  kernel_height * kernel_width
        # fan_out = output_channels * kernel_height * kernel_width
        std = np.sqrt(2 / (fan_in + fan_out))
        self.weights = np.random.normal(0, std, size=weights_shape)

        return self.weights


class He:

    def __init__(self):
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out=None):
        std = np.sqrt(2 / fan_in)
        self.weights = np.random.normal(0, std, size=weights_shape)

        return self.weights
