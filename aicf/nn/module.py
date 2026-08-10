from __future__ import annotations
from collections import OrderedDict
from typing import Iterator

from ..frontend.tensor import Parameter


class Module:
    """Base class for user-facing model components.

    The skeleton currently implements only the structural pieces needed for
    composing models: child-module registration and parameter registration.
    State dictionaries, buffers, train/eval mode and device movement are TODOs.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())

    def __setattr__(self, name, value):
        modules = self.__dict__.get("_modules")
        parameters = self.__dict__.get("_parameters")

        if modules is not None and name in modules:
            del modules[name]
        if parameters is not None and name in parameters:
            del parameters[name]

        if isinstance(value, Module):
            modules[name] = value
        elif isinstance(value, Parameter):
            parameters[name] = value

        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def children(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def named_children(self):
        return self._modules.items()

    def named_parameters(self, prefix: str = ""):
        for name, parameter in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, parameter

        for child_name, child in self._modules.items():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            yield from child.named_parameters(child_prefix)

    def parameters(self):
        for _, parameter in self.named_parameters():
            yield parameter

    def __repr__(self) -> str:
        if not self._modules:
            return f"{self.__class__.__name__}()"

        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            rendered = repr(module).replace("\n", "\n    ")
            lines.append(f"  ({name}): {rendered}")
        lines.append(")")
        return "\n".join(lines)
