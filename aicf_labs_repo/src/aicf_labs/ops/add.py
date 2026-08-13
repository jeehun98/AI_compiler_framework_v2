from __future__ import annotations

from .base import OperatorDefinition
from .registry import register


def _broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    lhs_rev = list(reversed(lhs))
    rhs_rev = list(reversed(rhs))
    width = max(len(lhs_rev), len(rhs_rev))

    for index in range(width):
        left = lhs_rev[index] if index < len(lhs_rev) else 1
        right = rhs_rev[index] if index < len(rhs_rev) else 1

        if left == right:
            result.append(left)
        elif left == 1:
            result.append(right)
        elif right == 1:
            result.append(left)
        else:
            raise ValueError(f"add shape mismatch: {lhs} + {rhs}")

    return tuple(reversed(result))


def _shape(inputs, attrs):
    lhs, rhs = inputs
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"add dtype mismatch: {lhs.dtype} + {rhs.dtype}")
    return _broadcast_shape(lhs.shape, rhs.shape)


ADD = register(OperatorDefinition("add", _shape))


def add(lhs, rhs):
    return ADD.emit((lhs, rhs))
