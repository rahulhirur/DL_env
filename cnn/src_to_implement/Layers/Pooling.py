import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):

        super().__init__()

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False
        
        self.pooling_output_width = None
        self.pooling_output_height = None
        
        self.hi_m = None
        self.wi_m = None
        
        self.is1dConv = 0

        self.input_tensor = None
        self.error_tensor = None
        

    def forward(self, input_tensor):
        
        self.input_tensor = input_tensor
        
        # Pooling layer must be implemented only for the 2D case
        if len(self.input_tensor.shape) <= 3:
            self.is1dConv = 1
        else:
            self.is1dConv = 0

        self.pooling_output_width = int(1 + (self.input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0])
        self.pooling_output_height = int(1 + (self.input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1])
        forward_output = np.zeros((input_tensor.shape[0], input_tensor.shape[1], self.pooling_output_height, self.pooling_output_width))

        for b in range(self.input_tensor.shape[0]):
            for c in range(self.input_tensor.shape[1]):
                for wi in range(self.pooling_output_width):
                    for hi in range(self.pooling_output_height):
                        forward_output[b, c, wi, hi] = np.max(self.input_tensor[b, c,
                                                              wi * self.stride_shape[0] : wi * self.stride_shape[0] + self.pooling_shape[0],
                                                              hi * self.stride_shape[1] : hi * self.stride_shape[1] + self.pooling_shape[1]])
        return forward_output
    

    def backward(self, error_tensor):
        
        """
        Derivative w.r.t. input has the same shape as the input
        """
        self.error_tensor = error_tensor
        backward_output = np.zeros_like(self.input_tensor)

        for b in range(self.input_tensor.shape[0]):
            for c in range(self.input_tensor.shape[1]):
                for wi in range(self.pooling_output_width):
                    for hi in range(self.pooling_output_height):
                        self.wi_m, self.hi_m = np.where(np.max(self.input_tensor[b, c, 
                                                                                 wi * self.stride_shape[0] : wi * self.stride_shape[0] + self.pooling_shape[0], 
                                                                                 hi * self.stride_shape[1] : hi * self.stride_shape[1] + self.pooling_shape[1]]) 
                                                        == self.input_tensor[b, c, 
                                                                             wi * self.stride_shape[0] : wi * self.stride_shape[0] + self.pooling_shape[0], 
                                                                             hi * self.stride_shape[1] : hi * self.stride_shape[1] + self.pooling_shape[1]])
                        self.wi_m, self.hi_m = self.wi_m[0], self.hi_m[0]
                        backward_output[b, c, 
                                        wi * self.stride_shape[0] : wi * self.stride_shape[0] + self.pooling_shape[0], 
                                        hi * self.stride_shape[1] : hi * self.stride_shape[1] + self.pooling_shape[1]][self.wi_m, self.hi_m] 
                        += error_tensor[b, c, wi, hi]

        return backward_output
