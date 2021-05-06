import torch
import torch.nn as nn
from torch.autograd import Variable


def get_rand(w, h, grad=True):
    t = torch.randn(w, h, requires_grad=grad)
    return nn.Parameter(Variable(t, requires_grad=grad))


class SubNet(nn.Module):
    def __init__(self, input_size, layers_dims=[50, 30]):
        super(SubNet, self).__init__()

        self.input_size = input_size
        self.layers_dims = [input_size + 0] + list(layers_dims) + [1]
        self.layers = []

        for i in range(1, len(self.layers_dims)):
            self.layers.append(nn.Linear(self.layers_dims[i-1], self.layers_dims[i]))
        self.net = nn.Sequential(*self.layers)

        self.kernel_weights = get_rand(1, input_size)

        self.encoder = None
        self.outputs = None

    def forward(self, x):
        for i in range(len(self.net) - 1):
            x = self.net[i](x)
            x = torch.tanh(x)
        x = self.net[-1](x)
        x = torch.sigmoid(x)

        self.outputs = x

        return x
