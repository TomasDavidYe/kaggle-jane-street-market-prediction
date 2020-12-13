import torch
import matplotlib.pyplot as plt

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

plt.scatter(x=t_u, y=t_c)
plt.show()


def normalize(x):
    return 0.1 * x


def model(x, w, b):
    return w * x + b


def loss_fn(y_1, y_2):
    squared_diffs = (y_1 - y_2) ** 2
    return squared_diffs.mean()


def dloss_fn(y_1, y_2):
    dsq_diffs = 2 * (y_1 - y_2) / y_1.size(0)
    return dsq_diffs


def dmodel_dw(x, w, b):
    return x


def dmodel_db(x, w, b):
    return 1.0


def grad_fn(x, y_true, y_pred, weights, biases):
    dloss_dtp = dloss_fn(y_pred, y_true)
    dloss_dw = dloss_dtp * dmodel_dw(x, weights, biases)
    dloss_db = dloss_dtp * dmodel_db(x, weights, biases)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


def training_loop(n_epochs, learning_rate, params, x, y_true):
    for epoch in range(1, n_epochs + 1):
        weights, biases = params
        y_pred = model(x, weights, biases)
        loss = loss_fn(y_true, y_pred)
        grad = grad_fn(x=x,
                       y_true=y_true,
                       y_pred=y_pred,
                       weights=weights,
                       biases=biases)
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        print(f'Params: {params}')
        print(f'Grad {grad}')
    return params


params = training_loop(n_epochs=5000,
                       learning_rate=1e-2,
                       params=torch.tensor([1.0, 0.0]),
                       x=normalize(t_u),
                       y_true=t_c)

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), model(normalize(t_u), *params).numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')

plt.show()
