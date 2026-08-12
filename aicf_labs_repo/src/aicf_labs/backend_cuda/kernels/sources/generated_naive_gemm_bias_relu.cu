extern "C" __global__ void kernel_0_fused_gemm_bias_relu(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C) {
  constexpr int M = 32;
  constexpr int N = 128;
  constexpr int K = 64;
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;
  constexpr int K_TILES = 2;

  for (int output = static_cast<int>(threadIdx.x);
       output < BM * BN;
       output += static_cast<int>(blockDim.x)) {
    const int local_m = output / BN;
    const int local_n = output % BN;
    const int row = static_cast<int>(blockIdx.y) * BM + local_m;
    const int col = static_cast<int>(blockIdx.x) * BN + local_n;
    if (row < M && col < N) {
      float acc = float(0);
      for (int kt = 0; kt < K_TILES; ++kt) {
        for (int kk = 0; kk < BK; ++kk) {
          const int k = kt * BK + kk;
          acc += A[row * K + k] * B[k * N + col];
        }
      }
      float value = acc + bias[col];
      value = value > float(0) ? value : float(0);
      C[row * N + col] = value;
    }
  }
}
