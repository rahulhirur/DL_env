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

        return np.subtract(weight_tensor, (self.learning_rate * gradient_tensor))


class SgdWithMomentum:
    
    def __init__(self, learning_rate, momentum_rate):
        
        try:
            if type(learning_rate) == int or type(learning_rate) == float:
                learning_rate = float(learning_rate)

            if type(momentum_rate) == int or type(momentum_rate) == float:
                momentum_rate = float(momentum_rate)

            self.learning_rate = learning_rate
            self.momentum_rate = momentum_rate

            self.prev_velocity = 0

            assert isinstance(self.learning_rate, float), "Error - Learning rate is not a number"
            assert isinstance(self.momentum_rate, float), "Error - Momentum rate is not a number"

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('//////')

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.prev_velocity = self.momentum_rate * self.prev_velocity - (self.learning_rate * gradient_tensor)
        
        return np.add(weight_tensor, self.prev_velocity)


class Adam:

    def __init__(self, learning_rate, mu, rho):
        try:
            if type(learning_rate) == int or type(learning_rate) == float:
                learning_rate = float(learning_rate)

            if type(mu) == int or type(mu) == float:
                mu = float(mu)

            if type(rho) == int or type(rho) == float:
                rho = float(rho)

            self.learning_rate = learning_rate
            self.mu = mu
            self.rho = rho

            self.prev_moment = 0
            self.prev_velocity = 0
            self.entry = 1

            assert isinstance(self.learning_rate, float), "Error - Learning rate is not a number"
            assert isinstance(self.mu, float), "Error - mu β1 is not a number"
            assert isinstance(self.rho, float), "Error - rho as β2 is not a number"

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('//////')

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.prev_velocity = (self.mu * self.prev_velocity) + ((1-self.mu) * gradient_tensor)
        self.prev_moment = (self.rho * self.prev_moment) + ((1 - self.rho) * (gradient_tensor * gradient_tensor))

        fin_v = (1 / (1-(self.mu ** self.entry))) * self.prev_velocity

        fin_u = (1 / (1-(self.rho ** self.entry))) * self.prev_moment

        self.entry = self.entry + 1

        return np.subtract(weight_tensor, (self.learning_rate * fin_v / (fin_u**0.5 + np.finfo(float).eps)))
