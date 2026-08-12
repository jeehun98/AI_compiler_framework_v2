from aicf_labs.backend_cuda import ObservationInputs, generated_naive_gemm_bias_relu, observe_kernel
from aicf_labs.backend_cuda.observation import observe_ast_dump, observe_ptx, observe_sass


def test_fixed_v020_kernel_is_observation_target():
    kernel = generated_naive_gemm_bias_relu()
    observation = observe_kernel(kernel)

    assert kernel.entry == "kernel_0_fused_gemm_bias_relu"
    assert observation.source is not None
    assert observation.source.metrics["global_kernels"] == 1
    assert observation.source.metrics["for_loops"] == 3
    assert observation.source.metrics["if_statements"] == 1
    assert observation.source.metrics["shared_declarations"] == 0


def test_artifact_observers_accept_already_collected_text():
    ast = observe_ast_dump("ForStmt\nIfStmt\nBinaryOperator\nBinaryOperator\n")
    ptx = observe_ptx("ld.global.f32 %f1, [%rd1];\nfma.rn.f32 %f3, %f1, %f2, %f3;\nst.global.f32 [%rd2], %f3;\n")
    sass = observe_sass("/*0000*/ LDG.E R2, [R4];\n/*0010*/ FFMA R6, R2, R3, R6;\n/*0020*/ STG.E [R8], R6;\n")

    assert ast.metrics["ForStmt"] == 1
    assert ast.metrics["BinaryOperator"] == 2
    assert ptx.metrics["ld_global"] == 1
    assert ptx.metrics["fma"] == 1
    assert sass.metrics["global_loads"] == 1
    assert sass.metrics["ffma"] == 1
