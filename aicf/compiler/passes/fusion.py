from ..pass_base import Pass
from ..analysis.use_def import build_use_def
from ...diagnostics.events import emit


class FusionPass(Pass):
    name = "fusion"

    def run(self, module, context):
        """Detect GEMM -> BiasAdd -> ReLU through actual IR dataflow.

        v0.5 only performs:
        1. candidate discovery
        2. legality checks
        3. decision logging

        Profitability analysis and IR rewriting are intentionally deferred.
        """

        analysis = build_use_def(module)

        candidate = self._find_candidate(module, analysis)

        if candidate is None:
            emit(
                context,
                "optimization.decision",
                {
                    "candidate": "gemm_bias_relu",
                    "found": False,
                    "legal": False,
                    "profitable": None,
                    "selected": False,
                    "reason": "connected dataflow pattern not found",
                },
            )
            return module

        gemm, bias_add, relu = candidate

        legal, reason = self._check_legality(
            gemm,
            bias_add,
            relu,
            analysis,
        )

        emit(
            context,
            "optimization.decision",
            {
                "candidate": "gemm_bias_relu",
                "found": True,
                "legal": legal,
                "profitable": None,
                "selected": False,
                "reason": reason,
            },
        )

        return module

    def _find_candidate(self, module, analysis):
        """Find a connected GEMM -> BiasAdd -> ReLU path.

        Operations do not need to be adjacent in module.ops.
        The relationship is discovered through IR uses.
        """

        for gemm in module.ops:
            if gemm.name != "gemm":
                continue

            if len(gemm.results) != 1:
                continue

            gemm_result = gemm.results[0]

            for gemm_use in analysis.uses(gemm_result):
                bias_add = gemm_use.user

                if bias_add.name != "bias_add":
                    continue

                # GEMM result must feed bias_add's x operand.
                if gemm_use.operand_index != 0:
                    continue

                if len(bias_add.results) != 1:
                    continue

                bias_result = bias_add.results[0]

                for bias_use in analysis.uses(bias_result):
                    relu = bias_use.user

                    if relu.name != "relu":
                        continue

                    # BiasAdd result must feed ReLU's input.
                    if bias_use.operand_index != 0:
                        continue

                    return gemm, bias_add, relu

        return None

    def _check_legality(
        self,
        gemm,
        bias_add,
        relu,
        analysis,
    ):
        """Check the minimal legality rules for this fusion."""

        gemm_result = gemm.results[0]
        bias_result = bias_add.results[0]

        if analysis.use_count(gemm_result) != 1:
            return (
                False,
                "gemm result has multiple uses",
            )

        if analysis.use_count(bias_result) != 1:
            return (
                False,
                "bias_add result has multiple uses",
            )

        # Defensive checks: candidate discovery already established these,
        # but legality should state its own invariants explicitly.
        if (
            not bias_add.operands
            or bias_add.operands[0] is not gemm_result
        ):
            return (
                False,
                "bias_add does not consume gemm result as operand 0",
            )

        if (
            not relu.operands
            or relu.operands[0] is not bias_result
        ):
            return (
                False,
                "relu does not consume bias_add result as operand 0",
            )

        return (
            True,
            "legal candidate; profitability and rewrite not implemented",
        )