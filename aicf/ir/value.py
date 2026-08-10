from __future__ import annotations
from dataclasses import dataclass
from .type import TensorType


@dataclass(eq=False)
class IRValue:
    """A value in compiler IR.

    IR values use object identity rather than structural equality. Two values
    may have the same name/type shape in malformed or intermediate IR but are
    still distinct objects from the compiler's point of view.
    """

    name: str
    type: TensorType