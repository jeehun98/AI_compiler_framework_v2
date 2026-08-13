from __future__ import annotations

from ..module import Module


class Sequential(Module):
    def __init__(self, *modules: Module):
        self.modules = tuple(modules)

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def __iter__(self):
        return iter(self.modules)
