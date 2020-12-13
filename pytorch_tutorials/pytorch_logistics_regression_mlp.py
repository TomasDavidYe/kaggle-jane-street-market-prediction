import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import torch


from deep_learning.MLP import MLP

NUM_OF_POINTS = 300
a = 2
b = 1
interval = np.arange(0, 2 * np.pi, 0.1)


def classify(x, y):
    return int(x ** 2 / a ** 2 + y ** 2 / b ** 2 < 1)


def generate_points(num_of_points):
    x_coordinates = (np.random.random(num_of_points) - 0.5) * 3 * a
    y_coordinates = (np.random.random(num_of_points) - 0.5) * 3 * b
    labels_local = (classify(x_coordinates[i], y_coordinates[i]) for i in range(num_of_points))
    return x_coordinates, y_coordinates, labels_local


ellipse_x = a * np.cos(interval)
ellipse_y = b * np.sin(interval)

x_points, y_points, labels = generate_points(num_of_points=NUM_OF_POINTS)
dataset = pd.DataFrame(data={
    'x': x_points,
    'y': y_points,
    'label': labels
})

dataset.plot.scatter(x='x', y='y', c='label', cmap=cm.get_cmap('Spectral'))
plt.plot(ellipse_x, ellipse_y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

mlp = MLP(3)

