import numpy as np

from Layers.Base import BaseLayer


class CrossEntropyLoss(BaseLayer):

    def __init__(self):
        super().__init__()
        self.label_tensor = None
        self.backward_output = None
        self.forward_output = None
        self.input_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # Compensate zero log output

        self.label_tensor = label_tensor
        self.input_tensor = np.finfo(float).eps + prediction_tensor
        self.forward_output = np.sum(-np.log(np.sum(self.input_tensor * label_tensor, axis=1)))

        return self.forward_output

    def backward(self, label_tensor):

        self.backward_output = - (label_tensor / self.input_tensor)
        return self.backward_output
