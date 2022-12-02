import numpy as np
from scipy import signal


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.stride_shape = tuple()
        self.input_channels = None
        self.kernel_height = None
        self.kernel_width = None
        self.convolution_shape = self.convolution_shape
        self.num_kernels = int()
        self.weights = np.random.rand(self.num_kernels,
                                      self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2])
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        for i in range(self.num_kernels):
            for j in range(self.convolution_shape[0]):

        pass

    def backward(self, error_tensor):
        pass
