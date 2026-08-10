from __future__ import annotations


class LoweringRegistry:
    """Map source IR operation names to target-specific lowering rules."""

    def __init__(self):
        self._rules = {}

    def register(self, op_name, fn=None):
        """Register a lowering function.

        Can be used directly or as a decorator.
        """

        def decorator(rule):
            if op_name in self._rules:
                raise ValueError(f"lowering rule already registered: {op_name}")
            self._rules[op_name] = rule
            return rule

        if fn is None:
            return decorator

        return decorator(fn)

    def get(self, op_name):
        return self._rules.get(op_name)

    def require(self, op_name):
        rule = self.get(op_name)
        if rule is None:
            raise KeyError(f"no lowering rule registered for op: {op_name}")
        return rule

    def registered_ops(self):
        return tuple(self._rules)