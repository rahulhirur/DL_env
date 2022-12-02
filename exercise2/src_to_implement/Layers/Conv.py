import numpy as np
from scipy import signal


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.input_height = None
        self.input_width = None
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape  # [c,m,n] = input_channels (c), spatial extent of kernel (m,n)
        self.num_kernels = num_kernels
        self.trainable = True
        self.input_channels = None
        self.kernel_height = None
        self.kernel_width = None
        self.weights = np.random.rand(self.num_kernels,
                                      self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2])
        self.input_tensor = None
        self.forward_output = None
        self.output_height = ((self.input_shape[1] - self.kernel_height) + 1) / self.stride_shape[0]
        self.output_width = ((self.input_shape[2] - self.kernel_width) + 1) / self.stride_shape[1]
        self.output_shape = num_kernels, self.output_height, self.output_width
        self.kernels_shape = num_kernels, self.input_channels, self.kernel_height, self.kernel_width
        self.input_shape = self.input_channels, self.input_height, self.input_width

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.forward_output = np.ones(self.output_shape)
        for i in range(self.num_kernels):
            for j in range(self.convolution_shape[0]):
                self.forward_output[i] += signal.correlate2d(self.input_tensor[j], self.weights[i, j], "valid")
        return self.forward_output

    def backward(self, error_tensor):
        gradient_weights = np.zeros(self.kernel_shape)
        backward_tensor = np.zeros(self.input_shape)
        for i in range(self.num_kernels):
            for j in range(self.convolution_shape[0]):
                gradient_weights[i, j] = signal.correlate2d(self.input_tensor[j], error_tensor[i], "valid")
                backward_tensor[j] += signal.convolve2d(error_tensor[i], self.weights[i, j], "full")
        self.weights -= self.learning_rate * gradient_weights
        self.biases -= self.learning_rate * error_tensor

        return backward_tensor
