import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import torch

from models.PythorchLogisticsRegression import PytorchLogisticsRegression
from visualisation.ClassifierVisualiser2D import ClassifierVisualiser2D

NUM_OF_POINTS = 100
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


def feature_transform(x, y):
    return torch.tensor([
        x,
        y,
        x ** 2,
        y ** 2
    ]).float()


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

train_target = torch.tensor(labels).float()
train_features = feature_transform(x=x_points, y=y_points)



model = PytorchLogisticsRegression(input_size=train_features.shape[0],
                                   learning_rate=0.5,
                                   n_epochs=1000)


before_train_prob = model.forward(train_features)
before_train_pred = (before_train_prob >= 0.5).float()

model.fit(x=train_features,
          y_true=train_target)
after_train_prob = model.forward(train_features)
after_train_pred = (after_train_prob >= 0.5).float()

print(f'Accuracy before: {100 * (train_target == before_train_pred).float().mean()}%')
print(f'Accuracy after: {100 * (train_target == after_train_pred).float().mean()}%')

ClassifierVisualiser2D().plot_decision_boundary(x=x_points,
                                                y=y_points,
                                                feature_transform=feature_transform,
                                                model=model)
