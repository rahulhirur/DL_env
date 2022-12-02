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
        self.forward_output = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.forward_output = np.ones(((self.convolution_shape[0] - self.kernel_height[0]) / self.stride_shape)+1)
        for i in range(self.num_kernels):
            for j in range(self.convolution_shape[0]):
                self.forward_output[i] = signal.correlate2d(self.input_tensor[j], self.weights[i,j], "same")
        return self.forward_output

    def backward(self, error_tensor):
        pass
