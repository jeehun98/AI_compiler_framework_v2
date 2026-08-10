from __future__ import annotations
from .module import Module


class Sequential(Module):
    """Apply child modules in declaration order."""

    def __init__(self, *modules: Module):
        super().__init__()
        for index, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError(
                    f"Sequential expects Module instances, got {type(module).__name__}"
                )
            setattr(self, str(index), module)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        return self._modules[str(index)]
