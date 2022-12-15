import numpy as np

from Layers.Base import BaseLayer


class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):

        super().__init__()

        self.pooling_output_width = None
        self.pooling_output_height = None
        self.hi_m = None
        self.wi_m = None
        self.trainable = False
        self.is1dConv = 0

        self.input_tensor = None
        self.error_tensor = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if len(self.input_tensor.shape) <= 3:
            self.is1dConv = 1
        else:
            self.is1dConv = 0

        self.pooling_output_height = int(
            1 + (self.input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0])
        self.pooling_output_width = int(1 + (self.input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1])
        forward_output = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
                                   self.pooling_output_height, self.pooling_output_width))

        for b in range(self.input_tensor.shape[0]):
            for c in range(self.input_tensor.shape[1]):
                for hi in range(self.pooling_output_height):
                    for wi in range(self.pooling_output_width):
                        forward_output[b, c, hi, wi] = np.max(self.input_tensor[b, c,
                                                              hi * self.stride_shape[0]: hi * self.stride_shape[0] +
                                                                                         self.pooling_shape[0],
                                                              wi * self.stride_shape[1]: wi * self.stride_shape[1] +
                                                                                         self.pooling_shape[1]])
        return forward_output

    def backward(self, error_tensor):
        """
        Derivative w.r.t. input has the same shape as the input
        :param error_tensor:
        :return:
        """
        self.error_tensor = error_tensor

        backward_output = np.zeros_like(self.input_tensor)

        for b in range(self.input_tensor.shape[0]):
            for c in range(self.input_tensor.shape[1]):
                for hi in range(self.pooling_output_height):
                    for wi in range(self.pooling_output_width):
                        self.hi_m, self.wi_m = np.where(np.max(self.input_tensor[b, c,
                                                               hi * self.stride_shape[0]: hi * self.stride_shape[0] +
                                                                                          self.pooling_shape[0],
                                                               wi * self.stride_shape[1]: wi * self.stride_shape[1] +
                                                                                          self.pooling_shape[1]])
                                                        == self.input_tensor[b, c,
                                                           hi * self.stride_shape[0]: hi * self.stride_shape[0] +
                                                                                      self.pooling_shape[0],
                                                           wi * self.stride_shape[1]: wi * self.stride_shape[1] +
                                                                                      self.pooling_shape[1]])
                        self.hi_m, self.wi_m = self.hi_m[0], self.wi_m[0]
                        backward_output[b, c,
                        hi * self.stride_shape[0]: hi * self.stride_shape[0] + self.pooling_shape[0],
                        wi * self.stride_shape[1]: wi * self.stride_shape[1] + self.pooling_shape[1]][
                            self.hi_m, self.wi_m] += error_tensor[b, c, hi, wi]

        return backward_output
