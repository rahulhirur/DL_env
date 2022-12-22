import numpy as np

from Layers.Base import BaseLayer


class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()
        self.forward_output = None
        self.backward_output = None

    def forward(self, input_tensor):
        p_exp = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.forward_output = p_exp / p_exp.sum(axis=1, keepdims=True)

        return self.forward_output

    def backward(self, error_tensor):

        self.backward_output = self.forward_output * (error_tensor - np.sum(error_tensor * self.forward_output, axis=1,
                                                                            keepdims=True))
        return self.backward_output
