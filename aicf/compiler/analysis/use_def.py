from __future__ import annotations
from dataclasses import dataclass, field

from ...ir.module import IRModule
from ...ir.operation import Operation
from ...ir.value import IRValue


@dataclass(frozen=True)
class Use:
    """One operand use of an IR value."""

    user: Operation
    operand_index: int


@dataclass
class ValueUseDef:
    """Producer and operand uses associated with one IR value."""

    producer: Operation | None = None
    uses: list[Use] = field(default_factory=list)


class UseDefAnalysis:
    """Read-only use-def information derived from an IRModule.

    The IR itself stays lightweight. Passes that need data-flow information can
    request this analysis rather than storing mutable producer/user state on
    every IRValue.
    """

    def __init__(
        self,
        info: dict[IRValue, ValueUseDef],
        op_indices: dict[Operation, int],
    ) -> None:
        self._info = info
        self._op_indices = op_indices

    def _entry(self, value: IRValue) -> ValueUseDef:
        try:
            return self._info[value]
        except KeyError as exc:
            raise KeyError(f"IR value {value.name} is not part of this analysis") from exc

    def producer(self, value: IRValue) -> Operation | None:
        return self._entry(value).producer

    def uses(self, value: IRValue) -> tuple[Use, ...]:
        return tuple(self._entry(value).uses)

    def users(self, value: IRValue) -> tuple[Operation, ...]:
        return tuple(use.user for use in self._entry(value).uses)

    def use_count(self, value: IRValue) -> int:
        return len(self._entry(value).uses)

    def unique_user_count(self, value: IRValue) -> int:
        return len({use.user for use in self._entry(value).uses})

    def op_index(self, op: Operation) -> int:
        try:
            return self._op_indices[op]
        except KeyError as exc:
            raise KeyError("operation is not part of this analysis") from exc


def build_use_def(module: IRModule) -> UseDefAnalysis:
    """Build producer/use relations for an IR module.

    Inputs and parameters are roots, so their producer is None. Operations are
    expected to be stored in topological order. A use records both the user
    operation and operand position so repeated uses such as add(%0, %0) remain
    distinguishable.
    """

    info: dict[IRValue, ValueUseDef] = {}
    op_indices = {op: index for index, op in enumerate(module.ops)}

    for value in [*module.inputs, *module.parameters]:
        if value in info:
            raise ValueError(f"IR root value {value.name} is registered more than once")
        info[value] = ValueUseDef(producer=None)

    for op in module.ops:
        for operand_index, operand in enumerate(op.operands):
            entry = info.get(operand)
            if entry is None:
                raise ValueError(
                    f"operation {op.name} uses undefined IR value {operand.name}"
                )
            entry.uses.append(Use(op, operand_index))

        for result in op.results:
            if result in info:
                raise ValueError(f"IR value {result.name} is defined more than once")
            info[result] = ValueUseDef(producer=op)

    for output in module.outputs:
        if output not in info:
            raise ValueError(f"IR output {output.name} is undefined")

    return UseDefAnalysis(info, op_indices)