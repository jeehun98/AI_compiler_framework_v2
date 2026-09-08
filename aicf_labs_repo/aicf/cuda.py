"""Selected implementation → launch boundary. No CUDA execution is claimed."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaImplementation:
    operator_type: str
    id: str

    def launch(self, inputs: tuple, attributes: dict) -> dict:
        # A future implementation replaces this method with build/launch logic.
        return {"backend": "cuda", "implementation_id": self.id,
                "status": "not-implemented", "output": None, "observations": []}


# Implementation identities, not another operator-semantics catalog.
IMPLEMENTATIONS = {
    kind: CudaImplementation(kind, f"cuda.{kind}.placeholder")
    for kind in ("matmul", "add", "mul", "relu", "reduce_sum", "linear_relu")
}


def lookup(operator_type: str) -> CudaImplementation:
    try:
        return IMPLEMENTATIONS[operator_type]
    except KeyError:
        raise ValueError(f"No CUDA implementation declared for {operator_type}") from None
