# CREATE RANDOM DATA POINTS
import torch
from matplotlib import pyplot as plt
import numpy as np

from data_generation.EllipseDatasetGenerator import EllipseDatasetGenerator
from visualisation.ClassifierVisualiser2D import ClassifierVisualiser2D


class Feedforward(torch.nn.Module):
    def __init__(self, input_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc1(x))

    def predict(self, x):
        prob = self.forward(x)
        return (prob >= 0.5).float()


def feature_transform(x, y):
    temp = torch.tensor([
        x,
        y
    ]).float()
    return temp.reshape(temp.size()[1], temp.size()[0])


def get_feature_size():
    return feature_transform(np.array([1, 1]), np.array([1, 2])).shape[1]


def run():
    ELLIPSE_A = 2
    ELLIPSE_B = 1
    NUM_OF_EPOCHS = 10
    LEARNING_RATE = 0.05

    TRAIN_SET_SIZE = 300
    TEST_SET_SIZE = 300

    ellipse = EllipseDatasetGenerator(a=ELLIPSE_A,
                                      b=ELLIPSE_B,
                                      interval_boundary=3)

    visualiser = ClassifierVisualiser2D(plt)

    x_coor_train, y_coor_train, labels_train = ellipse.generate_points(TRAIN_SET_SIZE)
    x_train = feature_transform(x_coor_train, y_coor_train)
    y_train = torch.FloatTensor(labels_train)

    x_coor_test, y_coor_test, labels_test = ellipse.generate_points(TEST_SET_SIZE)
    x_test = feature_transform(x_coor_test, y_coor_test)
    y_test = torch.FloatTensor(labels_test)

    model = Feedforward(get_feature_size())
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training', before_train.item())

    model.train()
    for NUM_OF_EPOCHS in range(NUM_OF_EPOCHS):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)

        print('Epoch {}: train loss: {}'.format(NUM_OF_EPOCHS, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()

    # model.fc1.weight = torch.nn.Parameter(torch.tensor([-0.0150,  0.6861, -1.3665, -5.5932]))
    # model.fc1.bias = torch.nn.Parameter(torch.tensor(5.2256))

    model.eval()
    y_pred = model(x_test)
    after_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss after Training', after_train.item())

    ellipse_x, ellipse_y = ellipse.get_curve()

    visualiser.plot_ellipse(x_points=x_coor_test,
                            y_points=y_coor_test,
                            labels=labels_test,
                            ellipse_x=ellipse_x,
                            ellipse_y=ellipse_y)

    visualiser.plot_decision_boundary(x=x_coor_test,
                                      y=y_coor_test,
                                      model=model,
                                      feature_transform=feature_transform)

    visualiser.show_plot()


run()
