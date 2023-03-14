import numpy as np


class Sgd:
    
    def __init__(self, learning_rate):
        
        try:
            if type(learning_rate) == int or type(learning_rate) == float:
                learning_rate = float(learning_rate)

            self.learning_rate = learning_rate

            assert isinstance(self.learning_rate, float), "Error - Learning rate is not a number"

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('//////')
            

    def calculate_update(self, weight_tensor, gradient_tensor):
        
        """"
        Returns the updated weights according to the basic gradient descent update scheme.
        """"
        
        return np.subtract(weight_tensor, (self.learning_rate * gradient_tensor))
