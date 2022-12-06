import numpy as np
from scipy import signal
import warnings

from Layers.Base import BaseLayer


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):

        super().__init__()

        self.pad_x = 0
        self.pad_y = 0
        self.as_x = 0
        self.as_y = 0

        self.trainable = True
        self.stride_shape = stride_shape

        self.convolution_shape = convolution_shape
        if len(self.convolution_shape) == 2:
            self.is1dConv = 1
        else:
            self.is1dConv = 0

        self.calc_padding()

        self.num_kernels = num_kernels

        self.weights = np.random.rand(self.num_kernels, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)

        self._gradient_weights = None
        self._gradient_bias = None

        self.input_tensor = None
        self.forward_output = None
        self.backward_output = None

        try:
            assert isinstance(self.num_kernels, int), "Error - Number of Kernels  is not an int"

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('//////')

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # Forward Output, o = ((i-k+2p)/s)+1
        # Shape of Forward Output = Batch Size, Number of Kernels, o_along_height, o_along_width
        if self.is1dConv:
            self.forward_output = np.zeros((input_tensor.shape[0],
                                            self.num_kernels,
                                            int(((input_tensor.shape[2] - self.convolution_shape[
                                                1] + 2 * self.pad_y + self.as_y) /
                                                 self.stride_shape[0]) + 1)))

        else:
            self.forward_output = np.zeros((input_tensor.shape[0],
                                            self.num_kernels,
                                            int(((input_tensor.shape[2 - self.is1dConv] - self.convolution_shape[
                                                1 - self.is1dConv] + 2 * self.pad_y + self.as_y) /
                                                 self.stride_shape[0]) + 1),
                                            int(((input_tensor.shape[3 - self.is1dConv] - self.convolution_shape[
                                                2 - self.is1dConv] + 2 * self.pad_x + self.as_x) /
                                                 self.stride_shape[1]) + 1)))

        input_tensor_shape = input_tensor.shape
        if self.is1dConv:
            if input_tensor_shape[2] - self.convolution_shape[0] <= 3:
                warnings.warn("Warning - Input is too small for the given convolution shape")

        else:
            if (input_tensor_shape[2] - self.convolution_shape[1] <= 3 or
                    input_tensor_shape[3] - self.convolution_shape[2] <= 3):
                warnings.warn("Warning - Input is too small for the given convolution shape")

        for i in range(input_tensor_shape[0]):
            for j in range(self.num_kernels):
                self.forward_output[i][j] = self.calculate_convolution(self.input_tensor[i], self.weights[j],
                                                                       self.bias[j])

        return self.forward_output

    def backward(self, error_tensor):
        # gradient_weights = np.zeros(self.kernel_shape)
        # backward_tensor = np.zeros(self.input_shape)
        # for i in range(self.num_kernels):
        #     for j in range(self.convolution_shape[0]):
        #         gradient_weights[i, j] = signal.correlate2d(self.input_tensor[j], error_tensor[i], "valid")
        #         backward_tensor[j] += signal.convolve2d(error_tensor[i], self.weights[i, j], "full")
        # self.weights -= self.learning_rate * gradient_weights
        # self.biases -= self.learning_rate * error_tensor

        # return backward_tensor
        pass

    def pad_2d(self, input_array, padded_value=0):

        return np.pad(input_array, ((self.pad_y, self.pad_y - self.as_y), (self.pad_x, self.pad_x - self.as_y)),
                      'constant', constant_values=padded_value)

    def calc_padding(self):
        if self.convolution_shape[1] == 1:
            self.pad_y = 0
            self.as_y = 0

        elif self.convolution_shape[1] % 2 == 0:
            self.pad_y = np.floor(self.convolution_shape[1] / 2)
            self.as_y = -1

        elif self.convolution_shape[1] % 2 != 0:
            self.pad_y = np.floor(self.convolution_shape[1] / 2)
            self.as_y = 0
        else:
            warnings.warn("Convolution shape is invalid. Defaulting to no padding in vertical direction")
            self.pad_y = 0
            self.as_y = 0

        if not self.is1dConv:

            if self.convolution_shape[2] == 1:
                self.pad_x = 0
                self.as_x = 0

            elif self.convolution_shape[2] % 2 == 0:
                self.pad_x = np.floor(self.convolution_shape[2] / 2)
                self.as_x = -1

            elif self.convolution_shape[2] % 2 != 0:
                self.pad_x = np.floor(self.convolution_shape[2] / 2)
                self.as_x = 0

            else:
                warnings.warn("Convolution shape is invalid. Defaulting to no padding in horizontal direction")
                self.pad_x = 0
                self.as_x = 0

    def calculate_convolution(self, input_vector, kernel_wt, kernel_bias):
        value = kernel_bias

        for i in range(0, input_vector.shape[0]):
            value += self.stride_convolve(input_vector[i], kernel_wt[i])

        return value

    def stride_convolve(self, input_vector, kernel_wt):
        if self.is1dConv:
            return signal.correlate(input_vector, kernel_wt, 'same', 'direct')[::self.stride_shape[0]]
        else:
            return signal.correlate2d(input_vector, kernel_wt, 'same', 'fill', 0)[:: self.stride_shape[0], :: self.stride_shape[1]]

