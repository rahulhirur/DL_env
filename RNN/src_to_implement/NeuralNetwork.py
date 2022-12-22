# Neural Network for CNN


import copy


class NeuralNetwork:
    # __init__ is the constructor class
    def __init__(self, optimizer_value, weights_initializer, bias_initializer):
        self.value_out = None
        self.value_in = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer_value
        self.loss = list()
        self.data_layer = None
        self.layers = list()
        self.loss_layer = None
        self._phase = None

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase_value):
        self._phase = phase_value

    def forward(self):
        self.value_in, self.value_out = self.data_layer.next()
        tmp_norm_sum = 0
        for layer in self.layers:
            layer.testing_phase = self.phase
            self.value_in = layer.forward(self.value_in)
            tmp_norm_sum += layer.norm_sum

        self.value_in = self.loss_layer.forward(self.value_in, self.value_out) + tmp_norm_sum

        self.loss.append(self.value_in)

        return self.value_in

    def backward(self):
        # value_out = self.value_out
        self.value_out = self.loss_layer.backward(self.value_out)
        for layer in reversed(self.layers):
            self.value_out = layer.backward(self.value_out)

    def append_layer(self, layer):

        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
# Initialize the weights and bias
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):

        self.phase = False

        for i in range(iterations):

            self.forward()
            self.backward()

    def test(self, input_tensor):

        self.phase = True
        value_in = input_tensor

        for layer in self.layers:
            value_in = layer.forward(value_in)

        return value_in

