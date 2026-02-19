import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(
            self, dim, out_dim=None, w=64, n_layers=2, time_varying=False,
            decoupled=False, sigmamin=0.0):
        super().__init__()
        self.time_varying = time_varying
        self.decoupled = decoupled
        self.sigmamin = sigmamin
        if out_dim is None:
            out_dim = dim
        self.net = nn.ModuleList()
        print(dim, time_varying, w, n_layers)
        print(type(dim), type(time_varying), type(w), type(n_layers))
        self.net.append(nn.Linear(dim + (1 if time_varying else 0), w))
        self.net.append(nn.SELU())

        for _ in range(n_layers):
            self.net.append(nn.Linear(w, w))
            self.net.append(nn.SELU())
        self.net.append(nn.Linear(w, out_dim))

    def forward(self, x):
        t, img = x[:, -1], x[:, :-1]
        for layer in self.net:
            x = layer(x)
        if self.decoupled:
            return (x - img) / (((1 - (1-self.sigmamin) * t)[:, None]))
        return x
