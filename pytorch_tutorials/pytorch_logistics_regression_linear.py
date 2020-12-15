import math
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import torch

from models.PythorchLogisticsRegression import PytorchLogisticsRegression

NUM_OF_POINTS = 300
a = 2
b = 1
interval = np.arange(0, 2 * np.pi, 0.1)


def classify(x, y):
    return int(x ** 2 / a ** 2 + y ** 2 / b ** 2 < 1)


def generate_points(num_of_points):
    x_coordinates = (np.random.random(num_of_points) - 0.5) * 3 * a
    y_coordinates = (np.random.random(num_of_points) - 0.5) * 3 * b
    labels_local = [classify(x_coordinates[i], y_coordinates[i]) for i in range(num_of_points)]
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

model = PytorchLogisticsRegression(input_size=4,
                                   learning_rate=0.5,
                                   n_epochs=1000)

train_target = torch.tensor(labels).float()
train_features = torch.tensor([
    x_points,
    y_points,
    x_points ** 2,
    y_points ** 2
]).float()




def get_meshgrid_axis(x, y, num_of_points=100):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1

    x_granularity = 10 ** int(math.log10(x_max - x_min)) / num_of_points
    y_granularity = 10 ** int(math.log10(y_max - y_min)) / num_of_points

    return np.arange(x_min, x_max, x_granularity), np.arange(y_min, y_max, y_granularity)


def get_meshgrid(x, y):
    x_axis, y_axis = get_meshgrid_axis(x, y)
    xx, yy = np.meshgrid(x_axis, y_axis)

    return xx, yy, x_axis, y_axis


def plot_decision_boundary():
    xx, yy, x_axis, y_axis = get_meshgrid(x_points, y_points)
    grid_features = torch.tensor([
        xx.ravel(),
        yy.ravel(),
        xx.ravel() ** 2,
        yy.ravel() ** 2
    ]).float()
    grid_predictions = model.predict(grid_features)
    colored_grid = grid_predictions.numpy().reshape(xx.shape)

    plt.contourf(xx, yy, colored_grid, alpha=0.4)
    plt.show()


before_train_prob = model.forward(train_features)
before_train_pred = (before_train_prob >= 0.5).float()

model.fit(x=train_features,
          y_true=train_target)
after_train_prob = model.forward(train_features)
after_train_pred = (after_train_prob >= 0.5).float()

print(f'Accuracy before: {100 * (train_target == before_train_pred).float().mean()}%')
print(f'Accuracy after: {100 * (train_target == after_train_pred).float().mean()}%')
plot_decision_boundary()
