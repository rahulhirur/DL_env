import numpy as np
from scipy import signal
import warnings
import copy

from Layers.Base import BaseLayer
from Optimization import Optimizers


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):

        super().__init__()

        self.norm_sum = None
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

        self.weights_shape = (self.num_kernels, *self.convolution_shape)
        self.bias_shape = (self.num_kernels, 1)

        self.weights = np.random.rand(self.num_kernels, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)
        if self.num_kernels == 1:
            self.bias = self.bias.reshape(1, 1)
        self._optimizer = None
        self._weight_optimizer = None
        self._bias_optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None

        self.input_tensor = None
        self.error_tensor = None
        self.forward_output = None
        self.backward_output = None

        try:
            assert isinstance(self.num_kernels, int), "Error - Number of Kernels  is not an int"

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('//////')

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_value):
        self._optimizer = optimizer_value
        self._weight_optimizer = copy.deepcopy(optimizer_value)
        self._bias_optimizer = copy.deepcopy(optimizer_value)

    @property
    def weight_optimizer(self):
        return self._weight_optimizer

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

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

        # rnn layer cnn layer
        self.input_tensor = input_tensor

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

        for i in range(input_tensor_shape[0]):
            for j in range(self.num_kernels):
                self.forward_output[i][j] = self.calculate_cross_correlation(self.input_tensor[i], self.weights[j],
                                                                             self.bias[j])

        return self.forward_output

    def backward(self, error_tensor):

        # self.norm_sum = self.weight_optimizer.regularizer.norm(self.weights)
        self.error_tensor = error_tensor
        self.backward_output = np.zeros((self.input_tensor.shape))

        error_updated = self.back_compatible_xgrad()

        self.gradient_weights = np.zeros((self.num_kernels, *self.convolution_shape))
        self.gradient_bias = np.zeros((self.num_kernels, 1))

        for i in range(error_tensor.shape[0]):  # Batch size
            for j in range(self.num_kernels):
                for k in range(self.input_tensor.shape[1]):
                    self.backward_output[i][k] += self.calculate_convolution(error_updated[i][j], self.weights[j][k])
                    self.gradient_weights[j][k] += self.back_correlate(self.pad_2d_backward(self.input_tensor[i][k]),
                                                                       error_updated[i][j])
                self.gradient_bias[j] += np.sum(error_updated[i][j])

        if self.optimizer is not None:
            self.weights = self.weight_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return self.backward_output

    def execute_padding(self, input_array, padded_value=0, pad_y=0, as_y=0, pad_x=0, as_x=0):
        if self.is1dConv:
            return np.pad(input_array, (pad_y, as_y), 'constant', constant_values=padded_value)

        else:
            return np.pad(input_array, ((pad_y, as_y), (pad_x, as_x)), 'constant', constant_values=padded_value)

    def pad_2d_backward(self, error_tensor, padded_value=0):

        pad_y, as_y, pad_x, as_x = self.calc_back_padding(self.convolution_shape[1:])

        return self.execute_padding(error_tensor, padded_value, pad_y, as_y, pad_x, as_x)

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
            # warnings.warn("Convolution shape is invalid. Defaulting to no padding in vertical direction")
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
                # warnings.warn("Convolution shape is invalid. Defaulting to no padding in horizontal direction")
                self.pad_x = 0
                self.as_x = 0

    def calc_back_padding(self, array_shape):

        pad_y = np.floor(array_shape[0] / 2)
        as_y = array_shape[0] - pad_y - 1

        if self.is1dConv:

            return int(pad_y), int(as_y), 0, 0
        else:

            pad_x = np.floor(array_shape[1] / 2)
            as_x = array_shape[1] - pad_x - 1

            return int(pad_y), int(as_y), int(pad_x), int(as_x)

    def back_compatible_xgrad(self):

        if self.is1dConv:
            error_updated = np.zeros((self.error_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2]))
            error_updated[:, :, ::self.stride_shape[0]] = self.error_tensor
        else:
            error_updated = np.zeros(
                (self.error_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]))
            error_updated[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = self.error_tensor
        return error_updated

    def calculate_cross_correlation(self, input_vector, kernel_wt, kernel_bias):

        value = 0
        # if self.is1dConv:
        #     value = np.full(input_vector.shape[1:], kernel_bias)[::self.stride_shape[0]]
        # else:
        #     value = np.full(input_vector.shape[1:], kernel_bias)[:: self.stride_shape[0], :: self.stride_shape[1]]

        for i in range(0, input_vector.shape[0]):
            value += self.stride_correlate(input_vector[i], kernel_wt[i])

        value = value + kernel_bias

        return value

    def calculate_convolution(self, input_vector, kernel_wt):

        value = self.stride_convolve(input_vector, kernel_wt)

        return value

    def stride_correlate(self, input_vector, kernel_wt):
        if self.is1dConv:
            return signal.correlate(input_vector, kernel_wt, 'same', 'direct')[::self.stride_shape[0]]
        else:
            return signal.correlate2d(input_vector, kernel_wt, 'same', 'fill', 0)[:: self.stride_shape[0],
                   :: self.stride_shape[1]]

    def back_correlate(self, input_vector, kernel_wt):
        if self.is1dConv:
            return signal.correlate(input_vector, kernel_wt, 'valid', 'direct')
        else:
            return signal.correlate2d(input_vector, kernel_wt, 'valid')

    def stride_convolve(self, input_vector, kernel_wt):

        if self.is1dConv:
            return signal.convolve(input_vector, kernel_wt, 'same', 'direct')
        else:
            return signal.convolve2d(input_vector, kernel_wt, 'same', 'fill', 0)

    def initialize(self, weights_initializer, bias_initializer):

        if self.is1dConv:
            fan_in = self.convolution_shape[0] * self.convolution_shape[1] * 1
            fan_out = self.num_kernels * self.convolution_shape[1] * 1

        else:
            fan_in = self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2]
            fan_out = self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2]

        self.weights = weights_initializer.initialize(self.weights_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias_shape, fan_in, fan_out)
