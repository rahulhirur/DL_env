import numpy as np
from Layers.Base import BaseLayer


class CrossEntropyLoss(BaseLayer):

    def __init__(self):
        
        super().__init__()
        
        self.input_tensor = None
        self.label_tensor = None
        
        self.forward_output = None
        self.backward_output = None
        

    def forward(self, prediction_tensor, label_tensor):
        
        """
        Computes the loss value as per CrossEntropy Loss formula accumulated over the batch
        """

        self.label_tensor = label_tensor
        
        # Epsilon is added to avoid log(0)=undefined
        self.input_tensor = np.finfo(float).eps + prediction_tensor 
        self.forward_output = np.sum(-np.log(np.sum(self.input_tensor * label_tensor, axis=1)))

        return self.forward_output
    

    def backward(self, label_tensor):
        
        """
        - The backpropagation starts here, hence no error tensor is needed
        - Instead, we need the label tensor
        """

        self.backward_output = - (label_tensor / self.input_tensor)
        
        return self.backward_output
