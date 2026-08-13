from ... import ops
from ..module import Module


class ReLU(Module):
    def forward(self, x):
        return ops.relu(x)
