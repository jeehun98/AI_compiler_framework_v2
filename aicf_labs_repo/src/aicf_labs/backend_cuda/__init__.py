"""Backend CUDA laboratory: fixed kernels, artifacts, and execution observation."""

from .kernels import KernelSpec, LaunchSpec, generated_naive_gemm_bias_relu
from .observation import ObservationInputs, observe_kernel
from .registry import ImplementationRegistry, default_registry

__all__ = [
    "ImplementationRegistry",
    "KernelSpec",
    "LaunchSpec",
    "ObservationInputs",
    "default_registry",
    "generated_naive_gemm_bias_relu",
    "observe_kernel",
]
