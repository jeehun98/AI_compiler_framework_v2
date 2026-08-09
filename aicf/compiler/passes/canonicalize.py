from ..pass_base import Pass


class CanonicalizePass(Pass):
    name = "canonicalize"

    def run(self, module, context):
        # TODO: canonical forms, algebraic simplification, constant folding.
        return module
