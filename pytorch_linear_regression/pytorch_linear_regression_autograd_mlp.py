import torch
import matplotlib.pyplot as plt

from deep_learning.MLP import MLP

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
mlp = MLP(5)
mlp.parameters()

plt.scatter(x=t_u, y=t_c)
plt.show()


def transform(x):
    return 0.1 * x


def loss_fn(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def training_loop(n_epochs, model: MLP, learning_rate, x, y_true):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x)
        # Compute Loss
        loss = loss_fn(y_true, y_pred)

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()


training_loop(n_epochs=5000,
              learning_rate=1e-4,
              model=mlp,
              x=t_u,
              y_true=t_c)

mlp.eval()
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), mlp.forward(t_u).detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')

plt.show()
