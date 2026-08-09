class LoweringRegistry:
    def __init__(self):
        self._rules = {}

    def register(self, op_name, fn):
        self._rules[op_name] = fn

    def get(self, op_name):
        return self._rules.get(op_name)
