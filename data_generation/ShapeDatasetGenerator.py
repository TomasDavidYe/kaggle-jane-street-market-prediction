from abc import ABC, abstractmethod
import numpy as np


class ShapeDatasetGenerator(ABC):
    def __init__(self, num_of_points, interval_boundary):
        self.__interval_boundary = interval_boundary
        self.__num_of_points = num_of_points

    @abstractmethod
    def classify(self, x, y):
        pass

    @abstractmethod
    def get_curve(self):
        pass


    def generate_points(self):
        x_coordinates = (np.random.random(self.__num_of_points) - 0.5) * self.__interval_boundary * 2
        y_coordinates = (np.random.random(self.__num_of_points) - 0.5) * self.__interval_boundary * 2
        labels_local = [self.classify(x_coordinates[i], y_coordinates[i]) for i in range(self.__num_of_points)]
        return x_coordinates, y_coordinates, labels_local
