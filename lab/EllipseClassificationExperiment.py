import torch
import matplotlib.pyplot as plt

from data_generation.EllipseDatasetGenerator import EllipseDatasetGenerator
from visualisation.ClassifierVisualiser2D import ClassifierVisualiser2D



class EllipseClassificationExperiment:
    def __init__(self, model, feature_transform, num_of_points=100, buffer_coefficient=3, ellipse_a=1, ellipse_b=1):
        self.__ellipse_a = ellipse_a
        self.__ellipse_b = ellipse_b
        self.__buffer_coefficient = buffer_coefficient
        self.__num_of_points = num_of_points
        self.__model = model
        self.__feature_transform = feature_transform

    def run(self):
        ellipse = EllipseDatasetGenerator(a=2, b=1, num_of_points=self.__num_of_points, interval_boundary=self.__buffer_coefficient)
        ellipse_x, ellipse_y = ellipse.get_curve()
        x_points, y_points, labels = ellipse.generate_points()

        train_target = torch.tensor(labels).float()
        train_features = self.__feature_transform(x_points, y_points)

        before_train_prob = self.__model.forward(train_features)
        before_train_pred = (before_train_prob >= 0.5).float()

        self.__model.fit(x=train_features,
                         y_true=train_target)

        after_train_prob = self.__model.forward(train_features)
        after_train_pred = (after_train_prob >= 0.5).float()

        print(f'Accuracy before: {100 * (train_target == before_train_pred).float().mean()}%')
        print(f'Accuracy after: {100 * (train_target == after_train_pred).float().mean()}%')

        visualiser = ClassifierVisualiser2D(plt)

        visualiser.plot_ellipse(x_points=x_points,
                                y_points=y_points,
                                labels=labels,
                                ellipse_x=ellipse_x,
                                ellipse_y=ellipse_y)

        visualiser.plot_decision_boundary(x=x_points,
                                          y=y_points,
                                          feature_transform=self.__feature_transform,
                                          model=self.__model)

        visualiser.show_plot()
