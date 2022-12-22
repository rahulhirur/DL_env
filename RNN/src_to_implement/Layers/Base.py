# Base Layer Creation
import numpy as np


class BaseLayer:

    def __init__(self):
        self.trainable = False
        self.testing_phase = False
        self.weights = np.array([])
        self.norm_sum = 0
