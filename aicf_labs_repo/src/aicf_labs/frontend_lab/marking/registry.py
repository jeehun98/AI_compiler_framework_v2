from __future__ import annotations

from .operator_mark import OperatorMark


class OperatorMarkRegistry:
    def __init__(self) -> None:
        self._marks: dict[str, OperatorMark] = {}

    def register(self, mark: OperatorMark) -> None:
        if not mark.operator:
            raise ValueError("operator name must not be empty")
        if mark.operator in self._marks:
            raise ValueError(f"operator mark already registered: {mark.operator}")
        self._marks[mark.operator] = mark

    def get(self, operator: str) -> OperatorMark:
        try:
            return self._marks[operator]
        except KeyError as exc:
            raise KeyError(f"operator mark not registered: {operator}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._marks)
