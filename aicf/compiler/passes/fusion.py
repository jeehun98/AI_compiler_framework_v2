from ..pass_base import Pass
from ...diagnostics.events import emit
from ...ir.operation import Operation


class FusionPass(Pass):
    name = "fusion"

    def run(self, module, context):
        """Fuse a legal GEMM -> BiasAdd -> ReLU dataflow chain.

        v0.6 deliberately keeps the policy simple:
        - discover a candidate through IR use-def
        - check minimal legality
        - select every legal candidate
        - rewrite the three operations into one fused operation

        Profitability/cost modeling is still intentionally not implemented.
        """
        analysis = context.analyses.use_def(module)
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

        if not legal:
            emit(
                context,
                "optimization.decision",
                {
                    "candidate": "gemm_bias_relu",
                    "found": True,
                    "legal": False,
                    "profitable": None,
                    "selected": False,
                    "reason": reason,
                },
            )
            return module

        self._rewrite(module, gemm, bias_add, relu)

        # The module was mutated in place, so any analysis built from the old
        # operation/value relationships is stale from this point onward.
        context.analyses.invalidate(module)

        emit(
            context,
            "optimization.decision",
            {
                "candidate": "gemm_bias_relu",
                "found": True,
                "legal": True,
                "profitable": None,
                "selected": True,
                "reason": "selected by fixed fusion policy; IR rewrite applied",
            },
        )

        return module

    def _find_candidate(self, module, analysis):
        """Find a connected GEMM -> BiasAdd -> ReLU dataflow path."""
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

                if gemm_use.operand_index != 0:
                    continue

                if len(bias_add.results) != 1:
                    continue

                bias_result = bias_add.results[0]

                for bias_use in analysis.uses(bias_result):
                    relu = bias_use.user

                    if relu.name != "relu":
                        continue

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
        """Check the minimal structural rules required by this rewrite."""
        if len(gemm.operands) != 2 or len(gemm.results) != 1:
            return False, "gemm arity is not supported by fusion"

        if len(bias_add.operands) != 2 or len(bias_add.results) != 1:
            return False, "bias_add arity is not supported by fusion"

        if len(relu.operands) != 1 or len(relu.results) != 1:
            return False, "relu arity is not supported by fusion"

        gemm_result = gemm.results[0]
        bias_result = bias_add.results[0]

        if analysis.use_count(gemm_result) != 1:
            return False, "gemm result has multiple uses"

        if analysis.use_count(bias_result) != 1:
            return False, "bias_add result has multiple uses"

        if bias_add.operands[0] is not gemm_result:
            return False, "bias_add does not consume gemm result as operand 0"

        if relu.operands[0] is not bias_result:
            return False, "relu does not consume bias_add result as operand 0"

        return True, "legal"

    def _rewrite(self, module, gemm, bias_add, relu) -> None:
        """Replace three operations with one fused operation.

        The fused operation reuses ReLU's result IRValue. This preserves all
        downstream references and module outputs without a separate replace-all-
        uses operation. The intermediate GEMM/BiasAdd values disappear together
        with their defining operations.
        """
        fused = Operation(
            name="fused_gemm_bias_relu",
            operands=[
                gemm.operands[0],
                gemm.operands[1],
                bias_add.operands[1],
            ],
            results=[relu.results[0]],
            attrs={
                "fused_ops": (
                    "gemm",
                    "bias_add",
                    "relu",
                )
            },
        )

        rewritten_ops = []

        for op in module.ops:
            if op is gemm or op is bias_add:
                continue

            if op is relu:
                # Insert at the old ReLU position so every fused operand that
                # was available to ReLU is still defined before the new op.
                rewritten_ops.append(fused)
                continue

            rewritten_ops.append(op)

        module.ops = rewritten_ops