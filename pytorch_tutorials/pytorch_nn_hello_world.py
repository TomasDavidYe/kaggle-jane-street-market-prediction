import torch
import torch.nn as nn
import matplotlib.pyplot as plt

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

plt.scatter(x=t_u, y=t_c)
plt.show()


def reshape(x):
    return x.reshape(x.shape[0], 1)


linear_model = nn.Linear(1, 1)
print(linear_model(reshape(t_u)))
