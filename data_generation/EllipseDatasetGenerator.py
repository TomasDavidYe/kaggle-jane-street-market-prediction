import numpy as np


# Notation as in https://en.wikipedia.org/wiki/Ellipse
from data_generation.ShapeDatasetGenerator import ShapeDatasetGenerator


class EllipseDatasetGenerator(ShapeDatasetGenerator):
    def __init__(self, a, b, interval_boundary):
        super().__init__(interval_boundary=interval_boundary)
        self.__a = a
        self.__b = b

    def get_curve(self):
        interval = np.arange(0, 2 * np.pi, 0.1)
        ellipse_x = self.__a * np.cos(interval)
        ellipse_y = self.__b * np.sin(interval)
        return ellipse_x, ellipse_y

    def classify(self, x, y):
        return int(x ** 2 / self.__a ** 2 + y ** 2 / self.__b ** 2 < 1)
