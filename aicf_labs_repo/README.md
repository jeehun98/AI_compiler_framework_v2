# AICF Labs

CUDA operator 구현을 하나씩 직접 작성하고 실행해 보는 최소 실험 저장소다.

현재는 FMA 하나만 다룬다. Frontend, backend registry, mask, AST, PTX, SASS
분석 구조는 사용하지 않는다.

## 구조

```text
aicf_labs_repo/
├─ README.md
└─ operators/
   └─ fma/
      ├─ fma.cu
      └─ run.ps1
```

새 operator가 필요해지면 `operators/<operator>/` 폴더를 하나 추가한다.

## FMA에서 비교하는 구현

분리 구현:

```text
temporary[i] = a[i] * b[i]
output[i] = temporary[i] + c[i]
```

최적화 구현:

```text
output[i] = fma(a[i], b[i], c[i])
```

최적화 커널은 FP16 값 두 개를 한 thread가 처리하도록 `half2`와
`__hfma2`를 사용한다. 분리 구현과 비교하면 다음 두 가지가 제거된다.

- FP16 intermediate tensor 하나
- Mul kernel과 Add kernel 중 한 번의 kernel launch

모든 배열은 연속 메모리이며 element 수는 짝수여야 한다. `cudaMalloc`으로
할당한 pointer는 `half2` 접근에 필요한 alignment를 만족한다.

이 코드는 elementwise CUDA FMA이며 Tensor Core MMA가 아니다.

## 환경과 실행

기준 환경:

- Windows
- CUDA Toolkit 13.3
- NVIDIA RTX 3060 (`sm_86`)
- Visual Studio 2022 C++ compiler

저장소 루트에서 실행한다.

```powershell
.\operators\fma\run.ps1
```

작은 입력으로 빠르게 확인하려면:

```powershell
.\operators\fma\run.ps1 -Elements 1048576 -Iterations 20
```

스크립트는 `fma.cu`를 executable 하나로 빌드한 뒤 분리 구현과 융합 구현의
평균 실행 시간을 출력한다. 생성 파일은 `operators/fma/build/` 아래에만
생기며 Git에 포함되지 않는다.
