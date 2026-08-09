from ..pass_base import Pass
from ...diagnostics.events import emit


class FusionPass(Pass):
    name = "fusion"

    def run(self, module, context):
        """Mock pass: detect GEMM -> BiasAdd -> ReLU only.

        It records a decision but intentionally does not rewrite IR yet.
        Actual legality/profitability/rewrite logic is a future implementation task.
        """
        names = [op.name for op in module.ops]
        candidate = any(names[i:i+3] == ["gemm", "bias_add", "relu"] for i in range(max(0, len(names)-2)))
        emit(context, "optimization.decision", {
            "candidate": "gemm_bias_relu",
            "found": candidate,
            "selected": False,
            "reason": "mock-only: rewrite not implemented" if candidate else "pattern not found",
        })
        return module
