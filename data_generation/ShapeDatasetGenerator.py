from abc import ABC, abstractmethod
import numpy as np


class ShapeDatasetGenerator(ABC):
    def __init__(self, interval_boundary):
        self.__interval_boundary = interval_boundary

    @abstractmethod
    def classify(self, x, y):
        pass

    @abstractmethod
    def get_curve(self):
        pass


    def generate_points(self, num_of_points):
        x_coordinates = (np.random.random(num_of_points) - 0.5) * self.__interval_boundary * 2
        y_coordinates = (np.random.random(num_of_points) - 0.5) * self.__interval_boundary * 2
        labels_local = [self.classify(x_coordinates[i], y_coordinates[i]) for i in range(num_of_points)]
        return x_coordinates, y_coordinates, labels_local
