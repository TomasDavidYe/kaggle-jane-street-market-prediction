import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)

plt.scatter(x=t_u, y=t_c)
plt.show()


def reshape(x):
    return x.reshape(x.shape[0], 1)


def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val,
                  t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)
        t_p_val = model(t_u_val)

        loss_val = loss_fn(t_p_val, t_c_val)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
              f" Validation loss {loss_val.item():.4f}")


t_un = 0.1 * t_u
t_cn = 0.1 * t_c

linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)
training_loop(
    n_epochs=1000,
    optimizer=optimizer,
    model=linear_model,
    loss_fn=nn.MSELoss(),
    t_u_train=t_un,
    t_u_val=t_un,
    t_c_train=t_cn,
    t_c_val=t_cn)

print()
print(linear_model.weight)
print(linear_model.bias)
