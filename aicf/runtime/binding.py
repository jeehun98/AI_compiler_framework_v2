from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frontend.tensor import (
    Parameter,
    TensorSpec,
    numpy_dtype,
    validate_host_array,
)


@dataclass(frozen=True)
class RuntimeValueSlot:
    """Compile-time description of one runtime-visible IR value."""

    ref: str
    name: str
    role: str
    spec: TensorSpec
    parameter: Parameter | None = None


@dataclass
class BoundBuffer:
    """Concrete host buffer bound to one IR value reference."""

    slot: RuntimeValueSlot
    array: np.ndarray

    @property
    def ref(self) -> str:
        return self.slot.ref

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)


@dataclass
class RuntimeBindings:
    """Concrete host buffers for one invocation of a compiled executable."""

    buffers: dict[str, BoundBuffer]

    def buffer(self, ref: str) -> BoundBuffer:
        try:
            return self.buffers[ref]
        except KeyError as exc:
            raise KeyError(f"no runtime buffer is bound for IR value {ref}") from exc

    def kernel_arguments(self, plan) -> tuple[np.ndarray, ...]:
        """Return host arrays in the exact argument order expected by a plan.

        For the current GEMM plans this naturally gives:
          plain GEMM:  A, B, C
          fused GEMM:  A, B, bias, C
        because lowering already records ordered input/output IR references.
        """

        refs = (*plan.inputs, *plan.outputs)
        return tuple(self.buffer(ref).array for ref in refs)

    def summary(self) -> list[dict[str, object]]:
        return [
            {
                "ref": bound.slot.ref,
                "name": bound.slot.name,
                "role": bound.slot.role,
                "shape": tuple(bound.array.shape),
                "dtype": bound.array.dtype.name,
                "nbytes": bound.nbytes,
            }
            for bound in self.buffers.values()
        ]


@dataclass(frozen=True)
class RuntimeSignature:
    """Runtime buffer contract derived from the optimized IR module."""

    inputs: tuple[RuntimeValueSlot, ...]
    parameters: tuple[RuntimeValueSlot, ...]
    results: tuple[RuntimeValueSlot, ...]

    @property
    def slots(self) -> tuple[RuntimeValueSlot, ...]:
        return (*self.inputs, *self.parameters, *self.results)

    def bind(self, args) -> RuntimeBindings:
        args = tuple(args)
        if len(args) != len(self.inputs):
            raise ValueError(
                f"expected {len(self.inputs)} runtime inputs, got {len(args)}"
            )

        buffers: dict[str, BoundBuffer] = {}

        for slot, value in zip(self.inputs, args):
            array = validate_host_array(value, slot.spec, copy=False)
            buffers[slot.ref] = BoundBuffer(slot, array)

        for slot in self.parameters:
            parameter = slot.parameter
            if parameter is None or parameter.data is None:
                raise ValueError(
                    f"parameter {slot.name} has no concrete host storage"
                )
            array = validate_host_array(
                parameter.data,
                slot.spec,
                copy=False,
            )
            buffers[slot.ref] = BoundBuffer(slot, array)

        for slot in self.results:
            array = np.zeros(
                slot.spec.shape,
                dtype=numpy_dtype(slot.spec.dtype),
            )
            buffers[slot.ref] = BoundBuffer(slot, array)

        return RuntimeBindings(buffers)


def _spec_from_ir_value(value, *, name: str) -> TensorSpec:
    return TensorSpec(
        shape=tuple(value.type.shape),
        dtype=value.type.dtype,
        name=name,
    )


def build_runtime_signature(
    module,
    input_specs,
    named_parameters,
) -> RuntimeSignature:
    """Build a simple one-buffer-per-IR-value runtime signature.

    v0.16 does not perform liveness analysis or buffer reuse. Every operation
    result receives its own concrete host buffer, including intermediates that
    are not module outputs. This is intentionally wasteful but makes every
    lowered kernel argument resolvable before memory optimization is added.
    """

    input_specs = tuple(input_specs)
    named_parameters = tuple(named_parameters)

    if len(module.inputs) != len(input_specs):
        raise ValueError(
            "runtime signature input count does not match optimized IR roots"
        )

    if len(module.parameters) != len(named_parameters):
        raise ValueError(
            "runtime signature parameter count does not match model parameters"
        )

    inputs = []
    for index, (value, declared_spec) in enumerate(
        zip(module.inputs, input_specs)
    ):
        name = declared_spec.name or f"input_{index}"
        spec = _spec_from_ir_value(value, name=name)
        inputs.append(
            RuntimeValueSlot(
                ref=value.name,
                name=name,
                role="input",
                spec=spec,
            )
        )

    parameters = []
    for value, (name, parameter) in zip(
        module.parameters,
        named_parameters,
    ):
        parameters.append(
            RuntimeValueSlot(
                ref=value.name,
                name=name,
                role="parameter",
                spec=_spec_from_ir_value(value, name=name),
                parameter=parameter,
            )
        )

    output_refs = {value.name for value in module.outputs}
    results = []
    for op in module.ops:
        for value in op.results:
            role = "output" if value.name in output_refs else "temporary"
            results.append(
                RuntimeValueSlot(
                    ref=value.name,
                    name=(
                        f"output_{value.name[1:]}"
                        if role == "output"
                        else f"temporary_{value.name[1:]}"
                    ),
                    role=role,
                    spec=_spec_from_ir_value(
                        value,
                        name=value.name,
                    ),
                )
            )

    return RuntimeSignature(
        inputs=tuple(inputs),
        parameters=tuple(parameters),
        results=tuple(results),
    )