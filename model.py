import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()
        self.n_class = n_class
        self.model = nn.Sequential(
            nn.Linear(10, 8),
            nn.Linear(8, 7),
            nn.Linear(7, n_class)
        )

    def remove_class(self):
        new_model = nn.Sequential(
            nn.Linear(10, 8),
            nn.Linear(8, 7),
            nn.Linear(7, self.n_class - 1)
        )

        for idx in range(len(self.model) - 1):
            new_model[idx].weight = nn.Parameter(self.model[idx].weight)
            new_model[idx].bias = nn.Parameter(self.model[idx].bias)

        old_weight = self.model[-1].weight
        new_weight = new_model[-1].weight
        old_bias = self.model[-1].bias
        new_bias = new_model[-1].bias

        for i in range(old_weight.shape[0]):
            pos = 0
            for j in range(old_weight.shape[1]):
                if i != self.n_class - 1:
                    with torch.no_grad():
                        new_weight[i][pos] = old_weight[i][j]
                        pos = pos + 1

        pos = 0
        for i in range(old_bias.shape[0]):
            if i != self.n_class - 1:
                with torch.no_grad():
                    new_bias[pos] = old_bias[i]
                    pos = pos + 1

        self.model[-1].weight = nn.Parameter(new_weight)
        self.model[-1].bias = nn.Parameter(new_bias)
        self.model = new_model
        self.n_class = self.n_class - 1

    def add_class(self):
        new_model = nn.Sequential(
            nn.Linear(10, 8),
            nn.Linear(8, 7),
            nn.Linear(7, self.n_class + 1)
        )
        for idx in range(len(self.model) - 1):
            new_model[idx].weight = nn.Parameter(self.model[idx].weight)
            new_model[idx].bias = nn.Parameter(self.model[idx].bias)

        old_weight = self.model[-1].weight
        new_weight = new_model[-1].weight
        old_bias = self.model[-1].bias
        new_bias = new_model[-1].bias

        for i in range(old_weight.shape[0]):
            for j in range(old_weight.shape[1]):
                with torch.no_grad():
                    new_weight[i][j] = old_weight[i][j]
        for i in range(old_bias.shape[0]):
            with torch.no_grad():
                new_bias[i] = old_bias[i]

        self.model[-1].weight = nn.Parameter(new_weight)
        self.model[-1].bias = nn.Parameter(new_bias)
        self.model = new_model
        self.n_class = self.n_class + 1

model = Net(5)
print(model)
model.remove_class()
print(model)
model.add_class()
print(model)