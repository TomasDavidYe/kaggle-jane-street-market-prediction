import torch
import numpy as np

from deep_learning.MLP import MLP
from lab.EllipseClassificationExperiment import EllipseClassificationExperiment


def feature_transform(x, y):
    return torch.tensor([
        x,
        y,
        x ** 2,
        y ** 2
    ]).float()


def get_feature_size():
    return feature_transform(np.array([1, 1]), np.array([1, 2])).shape[0]


model = MLP(input_size=get_feature_size(),
            learning_rate=0.5,
            n_epochs=1000)

EllipseClassificationExperiment(model=model,
                                feature_transform=feature_transform,
                                num_of_points=300,
                                buffer_coefficient=3,
                                ellipse_a=2,
                                ellipse_b=1).run()
