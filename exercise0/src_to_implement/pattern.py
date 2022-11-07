# Pattern File -
# Author - Rahul J Hirur

import numpy as np
import matplotlib.pyplot as plt


# Creating a Checker board Class

class Checker:

    # __init__ is the constructor class
    def __init__(self, resolution, tile):
        try:
            self.resolution = resolution
            self.tile = tile

            assert self.tile > 0, "Error - Tile size cannot be less than 1"
            assert self.resolution > 0, "Error - Resolution cannot be less than 1"
            assert self.resolution % (2 * self.tile) == 0, "Error - Tile size incorrect to make a Checkerboard"

            self.output = np.array([])

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('\n//////')

    def draw(self):
        b = np.ones([self.tile, self.tile])
        a = np.zeros([self.tile, self.tile])

        prime_matrix = np.concatenate([np.concatenate((a, b), axis=1), np.concatenate((b, a), axis=1)], axis=0)

        matrix_len = int(self.resolution / (2 * self.tile))
        outputs = np.tile(prime_matrix, [matrix_len, matrix_len])
        self.output = outputs * 1

        return outputs

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.colorbar()
        plt.show()


# Creating a Circle Class

class Circle:

    # __init__ is the constructor class
    def __init__(self, resolution, radius, position):
        try:

            self.resolution = resolution
            self.radius = radius
            self.position = position

            assert self.radius > 0, "Error - Radius cannot be less than 1"
            assert self.resolution > 0, "Error - Resolution cannot be less than 1"
            assert self.resolution >= (2 * self.radius), "Error - Circle Overflow"

            self.output = np.array([])

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('\n//////')

    def draw(self):
        lin_arr = np.linspace(1, self.resolution, self.resolution * 1)
        x, y = np.meshgrid(lin_arr, lin_arr)
        a = x * 0
        a[np.where((x - self.position[0] - 1) ** 2 + (y - self.position[1] - 1) ** 2 <= self.radius ** 2)] = 1
        b = a * 1

        a = a.astype('bool')
        self.output = b.astype('bool')

        return a

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.colorbar()
        plt.show()


# Creating a spectrum class
class Spectrum:

    # __init__ is the constructor class
    def __init__(self, resolution):
        try:
            self.resolution = resolution
            assert self.resolution > 0, "Error - Resolution cannot be less than 1"
            self.output = np.array([])

        except AssertionError as msg:
            print('\n//////')
            print(msg)
            print('\n//////')

    def draw(self):
        (template_mat, _) = np.meshgrid(np.linspace(0, 1, self.resolution), np.linspace(0, 1, self.resolution))

        red = template_mat
        blue = np.flip(red)
        green = np.rot90(blue)

        fin_out = np.dstack((red, green, blue))
        b = fin_out * 1

        self.output = b

        return fin_out

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.show()
