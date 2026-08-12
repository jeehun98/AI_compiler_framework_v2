from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


_NUMPY_DTYPES = {
    "float32": np.dtype(np.float32),
    "int32": np.dtype(np.int32),
}


def numpy_dtype(dtype: str) -> np.dtype:
    try:
        return _NUMPY_DTYPES[dtype]
    except KeyError as exc:
        raise NotImplementedError(
            f"host storage does not support dtype yet: {dtype}"
        ) from exc


def validate_host_array(
    value,
    spec: "TensorSpec",
    *,
    copy: bool = False,
) -> np.ndarray:
    """Validate a concrete host array against a compile-time TensorSpec.

    v0.16 deliberately keeps runtime binding strict: shape and dtype must match
    the compiled signature exactly. This avoids silently changing the program
    through implicit runtime casts while the execution/runtime layer is still
    being built.
    """

    array = np.asarray(value)
    expected_dtype = numpy_dtype(spec.dtype)

    if tuple(array.shape) != tuple(spec.shape):
        raise ValueError(
            f"host array shape mismatch for {spec.name or '<unnamed>'}: "
            f"expected {spec.shape}, got {tuple(array.shape)}"
        )

    if array.dtype != expected_dtype:
        raise TypeError(
            f"host array dtype mismatch for {spec.name or '<unnamed>'}: "
            f"expected {expected_dtype.name}, got {array.dtype.name}"
        )

    if copy:
        return np.array(array, copy=True)
    return array


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"
    name: Optional[str] = None


@dataclass
class Tensor:
    """Symbolic frontend tensor used while graph capture is active.

    `Tensor` deliberately remains a symbolic capture handle. v0.16 introduces
    concrete host storage for model Parameters and runtime-bound input/output
    buffers without mixing those runtime concerns into the symbolic frontend
    Tensor object.
    """

    spec: TensorSpec
    value: object | None = None


@dataclass(eq=False)
class Parameter:
    """Model-owned tensor state with minimal concrete host storage.

    Parameters now own a NumPy host array so a compiled executable can bind
    actual weight/bias buffers to generated kernel arguments. Initialization is
    intentionally simple and deterministic: absent explicit data, storage is
    zero-initialized. A framework-level initialization policy is deferred.
    """

    spec: TensorSpec
    requires_grad: bool = True
    data: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = np.zeros(
                self.spec.shape,
                dtype=numpy_dtype(self.spec.dtype),
            )
        else:
            self.data = validate_host_array(
                self.data,
                self.spec,
                copy=True,
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.shape

    @property
    def dtype(self) -> str:
        return self.spec.dtype

    def set_data(self, value) -> None:
        """Replace parameter storage after strict shape/dtype validation."""

        self.data = validate_host_array(
            value,
            self.spec,
            copy=True,
        )