import numpy as np

class Constant:
    def __init__(self, constant_value = 0.1):
        self.constant_value = constant_value
        self.kernel_height = None
        self.kernel_width = None

    def initialize(self, weights_shape, fan_in, fan_out):
        self.kernel_height, self.kernel_width = self.weights_shape
        self.weights = np.full((self.weights_shape), self.constant_value)
        self.weights = np.random.rand(self.weights_shape, self.fan_in, self.fan_in)

        return self.weights

class UniformRandom:

    def __init__(self):

        self.kernel_height = None
        self.kernel_width = None
        self.input_channels = None
        self.output_channels = None
        self.fan_in = None
        self.fan_out = None

    def initialize(self, weights_shape, fan_in, fan_out):

        self.kernel_height, self.kernel_width = self.weights_shape

        self.fan_in = self.input_channels * self.kernel_height * self.kernel_width
        self.fan_out = self.output_channels * self.kernel_height * self.kernel_width

        self.weights = np.random.rand(self.weight_shape, self.fan_in, self.fan_out)

        return self.weights

class Xavier:

    def __init__(self):

        self.kernel_height = None
        self.kernel_width = None
        self.input_channels = None
        self.output_channels = None
        self.fan_in = None
        self.fan_out = None

    def initialize(self, weights_shape, fan_in, fan_out):

        self.kernel_height, self.kernel_width = self.weights_shape

        self.fan_in = self.input_channels * self.kernel_height * self.kernel_width
        self.fan_out = self.output_channels * self.kernel_height * self.kernel_width
        self.std = np.sqrt(2/(self.fan_in + self.fan_out))

        self.weights = np.random.normal(0, self.std, size = self.weights_shape)

        return self.weights
        # self.weights = np.random.rand(self.weight_shape, self.fan_in, self.fan_out)

class He:

    def __init__(self):

        self.kernel_height = None
        self.kernel_width = None
        self.input_channels = None
        self.output_channels = None
        self.fan_in = None
        self.fan_out = None

    def initialize(self, weights_shape, fan_in, fan_out):

        self.kernel_height, self.kernel_width = self.weights_shape

        self.fan_in = self.input_channels * self.kernel_height * self.kernel_width
        self.fan_out = self.output_channels * self.kernel_height * self.kernel_width
        self.std = np.sqrt(2/(self.fan_in + self.fan_out))

        self.weights = np.random.normal(0, self.std, size = self.weights_shape)

        return self.weights