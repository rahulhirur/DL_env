# Optimizer File

import numpy as np


# Creating the SGD Class
class Optimizer:

    # __init__ is the constructor class
    def __init__(self, learning_rate=0.01):

        try:
            if type(learning_rate) == int or type(learning_rate) == float:
                learning_rate = float(learning_rate)

            self.learning_rate = learning_rate
            self.regularizer = None

            assert isinstance(self.learning_rate, float), "Error - Learning rate is not a number"

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('//////')

    # def calculate_update(self, weight_tensor, gradient_tensor):
    #
    #     return np.subtract(weight_tensor, (self.learning_rate * gradient_tensor))

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    # __init__ is the constructor class
    def __init__(self, learning_rate):

        super().__init__(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            return np.subtract(weight_tensor, (self.learning_rate * np.add(gradient_tensor, self.regularizer.calculate_gradient(weight_tensor))))
        else:
            return np.subtract(weight_tensor, (self.learning_rate * gradient_tensor))


# SgdWithMomentum
class SgdWithMomentum(Optimizer):
    # __init__ is the constructor class
    def __init__(self, learning_rate, momentum_rate):
        super().__init__(learning_rate)
        try:

            if type(momentum_rate) == int or type(momentum_rate) == float:
                momentum_rate = float(momentum_rate)

            self.momentum_rate = momentum_rate

            self.prev_velocity = 0

            assert isinstance(self.momentum_rate, float), "Error - Momentum rate is not a number"

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('//////')

    def calculate_update(self, weight_tensor, gradient_tensor):
        # new_velocity = mom_rate * prev_velocity - learning_rate * gradient_tensor
        # weight_tensor = weight_tensor - learning_rate * alpha * calculate_gradient(weight_tensor)
        # weight_tensor = weight_tensor + new_velocity
        # weight_tensor = weight_tensor - learning_rate * alpha * calculate_gradient(weight_tensor) + mom_rate * prev_velocity - learning_rate * gradient_tensor
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        self.prev_velocity = self.momentum_rate * self.prev_velocity - (self.learning_rate * gradient_tensor)


        # self.prev_velocity = self.momentum_rate * self.prev_velocity - (self.learning_rate * np.add(gradient_tensor, self.regularizer.calculate_gradient(weight_tensor)))
        return np.add(weight_tensor, self.prev_velocity)


# Adam Optimizers
class Adam(Optimizer):
    # __init__ is the constructor class
    def __init__(self, learning_rate, mu, rho):
        super().__init__(learning_rate)
        try:

            if type(mu) == int or type(mu) == float:
                mu = float(mu)

            if type(rho) == int or type(rho) == float:
                rho = float(rho)

            self.mu = mu
            self.rho = rho

            self.prev_moment = 0
            self.prev_velocity = 0
            self.entry = 1

            assert isinstance(self.mu, float), "Error - mu β1 is not a number"
            assert isinstance(self.rho, float), "Error - rho as β2 is not a number"

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('//////')

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.prev_velocity = (self.mu * self.prev_velocity) + ((1 - self.mu) * gradient_tensor)
        self.prev_moment = (self.rho * self.prev_moment) + ((1 - self.rho) * (gradient_tensor * gradient_tensor))

        fin_v = (1 / (1 - (self.mu ** self.entry))) * self.prev_velocity

        fin_u = (1 / (1 - (self.rho ** self.entry))) * self.prev_moment

        self.entry = self.entry + 1
        if self.regularizer is not None:
            return np.subtract(weight_tensor, (self.learning_rate * np.add(fin_v / (fin_u ** 0.5 + np.finfo(float).eps), self.regularizer.calculate_gradient(weight_tensor))))
        else:
            return np.subtract(weight_tensor, (self.learning_rate * fin_v / (fin_u ** 0.5 + np.finfo(float).eps)))
