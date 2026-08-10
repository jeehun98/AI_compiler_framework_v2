from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..frontend.tensor import TensorSpec

if TYPE_CHECKING:
    from .node import Node


@dataclass(eq=False)
class Value:
    """A symbolic value flowing through the captured graph.

    `producer` points to the node that defines this value. Graph inputs and
    parameters have no producer. `users` records one entry for each operand
    use, so the same node may appear more than once if it consumes the value
    multiple times.
    """

    id: int
    spec: TensorSpec
    name: str | None = None

    producer: Node | None = field(
        default=None,
        repr=False,
    )

    users: list[Node] = field(
        default_factory=list,
        repr=False,
    )