from exercise1.src_to_implement.Layers.FullyConnected import FullyConnected
import numpy as np

input_value = np.array(([1, 0],[0, 0],[1,1]))
label_value = np.array(([1,0],[0,1],[0,1]))


a = FullyConnected(2, 3)


