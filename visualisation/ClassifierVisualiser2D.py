import numpy as np
import pandas as pd
from matplotlib import cm
import math


class ClassifierVisualiser2D:
    def __init__(self, plot):
        self.__plot = plot

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

        self.__plot.contourf(xx, yy, colored_grid, alpha=0.4)



    def show_plot(self):
        self.__plot.show()

    def plot_ellipse(self, x_points, y_points, labels, ellipse_x, ellipse_y):
        dataset = pd.DataFrame(data={
            'x': x_points,
            'y': y_points,
            'label': labels
        })

        dataset.plot.scatter(x='x', y='y', c='label', cmap=cm.get_cmap('Spectral'))
        self.__plot.plot(ellipse_x, ellipse_y)
        self.__plot.xlabel("X")
        self.__plot.ylabel("Y")


