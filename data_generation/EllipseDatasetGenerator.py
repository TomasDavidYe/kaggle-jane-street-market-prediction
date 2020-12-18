import numpy as np


# Notation as in https://en.wikipedia.org/wiki/Ellipse
class EllipseDatasetGenerator:
    def __init__(self, a, b, ):
        self.__a = a
        self.__b = b

    def classify(self, x, y):
        return int(x ** 2 / self.__a ** 2 + y ** 2 / self.__b ** 2 < 1)

    def generate_points_around_ellipse(self, num_of_points, buffer_coefficient):
        x_coordinates = (np.random.random(num_of_points) - 0.5) * buffer_coefficient * self.__a
        y_coordinates = (np.random.random(num_of_points) - 0.5) * buffer_coefficient * self.__b
        labels_local = [self.classify(x_coordinates[i], y_coordinates[i]) for i in range(num_of_points)]
        return x_coordinates, y_coordinates, labels_local

    def get_ellipse_curve(self):
        interval = np.arange(0, 2 * np.pi, 0.1)
        ellipse_x = self.__a * np.cos(interval)
        ellipse_y = self.__b * np.sin(interval)
        return ellipse_x, ellipse_y
