import matplotlib.pyplot as plt
import numpy as np
import math


class ClassifierVisualiser2D:
    def __init__(self):
        pass

    def get_meshgrid_axis(self, x, y, num_of_points=100):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1

        x_granularity = 10 ** int(math.log10(x_max - x_min)) / num_of_points
        y_granularity = 10 ** int(math.log10(y_max - y_min)) / num_of_points

        return np.arange(x_min, x_max, x_granularity), np.arange(y_min, y_max, y_granularity)

    def get_meshgrid(self, x, y):
        x_axis, y_axis = self.get_meshgrid_axis(x, y)
        xx, yy = np.meshgrid(x_axis, y_axis)

        return xx, yy, x_axis, y_axis

    def plot_decision_boundary(self, x, y, feature_transform, model):
        xx, yy, x_axis, y_axis = self.get_meshgrid(x, y)
        grid_features = feature_transform(xx.ravel(), yy.ravel())
        grid_predictions = model.predict(grid_features)
        colored_grid = grid_predictions.numpy().reshape(xx.shape)

        plt.contourf(xx, yy, colored_grid, alpha=0.4)
        plt.show()
