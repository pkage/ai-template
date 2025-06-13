from torch import nn, einsum
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        # complete examples
        dim_input = 64
        dim_hidden = 128
        dim_output = 10

        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_output)

    def forward(self, x, *, rank, world_size):
        # do some training...
        x = F.relu(self.fc1(x))
        return self.fc2(x)
