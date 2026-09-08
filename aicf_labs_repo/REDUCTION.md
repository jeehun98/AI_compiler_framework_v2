# 최소 구조로 축소 — 삭제 전 inventory와 설계

2026-09-08. 이 문서는 기존 코드를 삭제하기 전에 작성했다.

## Phase 1: 현재 구조와 분류

기준: HEAD `2c93d0d08d070fd79781500ad88ee6b756cbb2ff`와 직전 미커밋 변경.
Git tracked + untracked/non-ignored 파일 180개, 코드 파일 159개,
코드 25,897줄. 코드 확장자는 `.py/.ts/.tsx/.cu/.hpp/.ps1/.css/.html`이며
빈 줄·주석을 포함한다. 의존성, 캐시, 빌드 산출물은 측정에서 제외한다.

| 영역 | 파일 | 코드 줄 | 분류 | 이유 / 처리 |
| --- | ---: | ---: | --- | --- |
| `python/aicf_labs/` 및 Python 문서 | 27 | 2,703 | SIMPLIFY | model/graph/reference 실행 아이디어만 작은 Python package로 통합 |
| `knowledge/` | 5 | — | MERGE | 관계 DSL·JSON schema·index를 제거하고 필요한 property만 Python에 정의 |
| `operators/` | 40 | 8,913 | DELETE | 29개 CUDA 실험 폴더와 benchmark/validation; 선택→CUDA interface만 새 구조에 남김 |
| `tools/` | 17 | 1,586 | DELETE | build/observation/SASS/profiler/knowledge pipeline은 최소 경로에 미연결 |
| `fixtures/` | 1 | — | DELETE | Python↔Web 양쪽 rewrite용 fixture 대신 실제 backend report 사용 |
| `tests/` | 14 | 2,101 | SIMPLIFY | 단일 경로와 대표 legality 경계만 테스트 |
| `web/` | 74 | 10,594 | SIMPLIFY | 최소 viewer와 Vite/TypeScript만 유지; TS에서 변환하지 않음 |
| README / ignore | 2 | — | SIMPLIFY | 실행 방법과 4개 핵심 Python 모듈 위주로 재작성 |

기존 전체 authored directory 구조:

```text
aicf_labs_repo/
  fixtures/
  knowledge/schemas/
  operators/{abs,add,batched_gemm,clamp,conv2d,copy,div,exp,fma,gemm,
    layernorm,log,max,min,mul,neg,reduce_max,reduce_mean,reduce_min,
    reduce_sum,relu,rmsnorm,sigmoid,softmax,sqrt,sub,tanh,transpose,where}/
  python/aicf_labs/{layers,operators}/
  tests/
  tools/{cuda_artifacts,cuda_runtime,knowledge,operator}/
  web/src/{backend,catalog,components,core,domain,examples,store,test,workspaces}/
```

### 확인한 실제 dependency와 중복

* Web `App.tsx` → Zustand `graphStore.ts` → TS rewrite engine/catalog/graph
  codec. Model/Graph/Kernel/Runtime/Hardware 다섯 workspace를 App이 import한다.
* Kernel/Runtime/Hardware는 placeholder kernel/timeline/관측값 중심이다.
  해당 화면, layer observation, probe context, Explorer 편집 모드는 DELETE.
* `web/src/backend/backendAdapter.ts`는 실제 Python/CUDA 실행 없이 인터페이스만
  정의한다. DELETE; viewer는 Python report만 읽는다.
* Python `__init__.py`가 frontend, masks, tracing, trace registry, selection,
  verification을 넓게 노출한다. `frontend.py`는 Model/Operator와 연결되지만
  Web 실행 경로와 독립이다. trace/planned unit/binding/evidence의 여러 value
  object를 최소 run report와 CUDA selection으로 MERGE한다.
* Operator 의미는 knowledge JSON, Python metadata, Web catalog에 분산되고,
  직전 공통 catalog도 별도 타입·validator·projection을 요구했다. 실제 필요한
  property를 `aicf/operators.py`에만 두고 Web에는 report로 전달한다.
* SASS/profiler code가 완전히 사용되지 않는다고 단정하지 않는다. 독립 실험
  workflow에서는 사용되지만, 이번에 남길 모델→변환→선택 경로에는 필요 없다.
* React Flow, KaTeX, Zustand, 복잡한 store/codec/rule/candidate UI, 관련 테스트는
  DELETE. React/Vite/TypeScript 실행 도구만 KEEP한다.

### 복구

기존 tracked 구현은 위 Git commit에서 복구할 수 있다. 직전 미커밋 소스까지
포함한 180개 authored 파일은 현재 프로젝트 밖에 백업했다:

`C:/Users/as042/.codex/visualizations/2026/09/07/01a0793e-d3a9-7232-88d0-8a4e30ff29ac/pre-reduction/source.zip`

같은 위치의 `inventory.json`에 전체 파일 목록/측정값이 있다. 백업에는 ignored
CUDA binary/artifact/runtime report, node_modules와 캐시는 포함하지 않는다.
현재 tree 안에는 archive/legacy/wrapper 구조를 남기지 않는다.

## Phase 2: 최소 구조 제안

```text
aicf_labs_repo/
  aicf/
    __init__.py
    model.py       # Sequential/Linear/ReLU → Graph
    operators.py   # operator instance, graph, 유일한 semantics 정의
    backend.py     # 두 rule, legality, reference 실행, 변환/선택
    cuda.py        # implementation ID lookup + 명시적 미구현 launch interface
  demo.py          # 모델 → 변환 → reference 비교 → CUDA 선택 → JSON report
  tests/test_pipeline.py
  web/
    src/{App.tsx,App.test.tsx,report.ts,main.tsx,style.css}
    public/report.json  # demo.py가 생성; hand-maintained catalog가 아님
    package.json, package-lock.json, index.html, TypeScript/Vite 설정
  README.md
  REDUCTION.md
```

1. **Source of truth:** `operators.py`의 작은 property 목록. 입력별 LINEAR,
   비음수 scalar 조건, PURE/ELEMENTWISE/ASSOCIATIVE/REDUCTION만 필요한 만큼
   표현한다. 별도 관계 DSL/schema/capability engine은 제거한다.
2. **Model → Graph:** 명시적 weight/bias를 가진 `Sequential(Linear, ReLU,
   Linear)`가 MatMul/Add/ReLU 인스턴스와 연결을 생성한다. eager tracing,
   SSA, Tensor class, registry는 필요 없다.
3. **Backend rules:** `fuse_linear_relu`와 `propagate_positive_scale` 두 기존
   변환만 Python 함수로 남긴다. 구조 탐색/조건 검사/새 그래프 생성이 한 파일에
   있다. 조건 부족은 UNKNOWN, 위반은 REJECT, 적용 가능은 ACCEPT이다.
4. **실행:** 작은 reference evaluator로 변환 전후 결과를 비교한다. 샘플 결과
   비교를 모든 입력의 IEEE 동치 증명으로 취급하지 않는다.
5. **선택 → CUDA:** backend가 graph의 계산 노드별 implementation을 선택하고
   `cuda.py` lookup/launch를 호출한다. 실제 CUDA 구현이 없으므로 launch 결과는
   명시적으로 `not-implemented`, output은 null, observations는 비어 있다.
6. **Web:** Python 실행으로 생성한 report를 보여주는 단일 viewer. 모델 목록,
   원본/변환 graph의 노드·edge, 선택 operator의 semantics, 변환 결과, 출력
   비교와 CUDA implementation ID만 표시한다. 편집/TS 규칙/가짜 측정 없음.
7. **테스트:** 모델 생성, 의미, 두 backend rule의 조건/결과, reference 비교,
   CUDA 선택/미구현 interface, 실제 report의 최소 Web 렌더링을 확인한다.

이 제안을 기준으로 삭제/통합을 진행한다. 최종 측정과 검증 결과는 아래에 추가한다.

## Phase 3–5: 실제 축소 및 검증 결과

### 삭제와 통합

최소 대체 경로를 먼저 완성하여 Python 8개 테스트, Web 2개 테스트와 빌드를
통과시킨 뒤 레거시 파일을 제거했다. 첫 재귀 디렉터리 삭제 요청은 자동 승인
검토에서 단계적 대체가 더 안전하다는 이유로 거부됐다. 검증 완료 후에는 삭제
대상 431개 파일을 `verified-removal.zip`에 모두 보관하고 SHA-256 일치를
확인했다. 이 목록의 파일만 개별 삭제하는 작업은 승인되어 완료됐다.

두 백업은 현재 프로젝트 밖의 위 `pre-reduction/` 디렉터리에 있다:

* `source.zip`: 작업 시작 시 authored 파일 180개. 직전 미커밋 semantic 작업 포함.
* `verified-removal.zip`: 실제 제거한 레거시 파일과 ignored CUDA 산출물 포함.
  `verified-removal.json`에 삭제 경로와 SHA-256이 있다.

현재 tree에 archive, legacy wrapper, deprecated alias는 없다. 옛 API와 문서,
예제와 해당 테스트도 제거했으며 API 호환성을 유지하는 변경이 아니다.

통합 결과:

| 기존 | 최종 |
| --- | --- |
| Python frontend/layer/operator/mask/trace/plan/selection/verification 계층 | `aicf/model.py`, `operators.py`, `backend.py`, `cuda.py` |
| Knowledge schema/index + 공통 관계 catalog + Python/Web metadata | `operators.py`의 `SEMANTICS` 하나 |
| Web rule/capability/matcher/legality engine | Python `backend.py`의 함수 두 개 |
| CUDA operators/tools/SASS/profiling 및 implementation evidence | `cuda.py`의 ID lookup + 미구현 launch 결과 |
| 다섯 workspace와 편집/탐색/store/candidate panel | 보고서 전용 `App.tsx` |

### 최종 전체 authored tree

```text
aicf_labs_repo/
  .gitignore
  README.md
  REDUCTION.md
  aicf/
    __init__.py
    model.py
    operators.py
    backend.py
    cuda.py
  demo.py
  tests/test_pipeline.py
  web/
    .gitignore
    index.html
    package.json
    package-lock.json
    tsconfig.json
    vite.config.ts
    src/
      main.tsx
      App.tsx
      App.test.tsx
      report.ts
      style.css
```

`web/public/report.json`은 실행 시 생성되며 Git에서 제외한다. Web의 `predev`,
`pretest`, `pretypecheck`, `prebuild`가 `demo.py`를 먼저 실행하므로 별도로
유지하는 JSON catalog나 mock fixture가 없다. 의존성/빌드/캐시는 위 tree에서
제외했다. Web runtime dependency도 React/ReactDOM 두 개만 남겼다.

### 실제 end-to-end 경로

1. `demo.py`가 `Sequential(Linear, ReLU, Linear)`를 정의한다.
2. `model.py`가 input/weight/bias와 MatMul/Add/ReLU 노드를 연결한다.
3. 각 `Operator.semantics`가 `operators.py`의 단일 정의를 조회한다.
4. `backend.transform`이 private intermediate와 [N] bias를 검사하고
   MatMul→Add→ReLU를 LinearReLU로 대체한다. 원본 graph는 보존한다.
5. 같은 입력 `[[1,2],[3,4]]`에서 reference 출력은 양쪽 모두 `[[3],[3]]`다.
6. `select_implementations`가 각 계산 노드에서 `cuda.lookup`을 호출한다.
   선택 ID는 `cuda.linear_relu.placeholder`, `cuda.matmul.placeholder`,
   `cuda.add.placeholder`다.
7. 실제로 각 implementation의 `launch` 메서드까지 호출한다. CUDA는 아직
   구현되지 않아 `not-implemented`, null output, 빈 observations를 반환한다.
   이 결과를 reference 실행 결과와 분리해서 보고서에 기록한다.
8. Web이 보고서의 model/graph/edge/semantics/decision/출력/CUDA ID를 표시한다.

남긴 규칙은 `fuse_linear_relu`, `propagate_positive_scale` 두 개다. 별도 Rule
base class나 registry가 없고, `transform`에서 명시적으로 선택한다. 첫 규칙은
기본 모델에서 실제 사용한다. 두 번째 규칙은 비음수 동차성 조건을 읽고,
양수/0 승인, 음수 거부, 미선언 property와 미증명 실행 도메인 UNKNOWN을
핵심 테스트에서 확인한다. 새 수학적 relation은 추가하지 않았다.

### Validation

삭제 후 최종 트리에서 실행:

| 검사 | 결과 |
| --- | --- |
| `python -m unittest discover -s tests -v` | 8 tests passed, skip 없음 |
| `npm test` | 2 viewer tests passed; Python report를 생성하여 소비 |
| `npm run typecheck` | passed |
| `npm run build` | passed; JS 약 199.47 kB, CSS 약 1.72 kB |
| `git diff --check` | passed |

테스트는 모델→graph, 입력별 의미, match/check/apply, 원본 불변성, 여러 reference
입력의 출력 일치, 공유 intermediate 거부, 동적 bias UNKNOWN, 양수/0/음수
scale, IEEE 요청 UNKNOWN, 잘못된 shape/graph, CUDA lookup과 null 출력,
실제 report 렌더링과 노드 선택에 한정한다. 제거한 기능의 테스트는 남기지 않았다.

### 크기 변화

삭제 전과 같은 측정 기준을 사용했다. Git tracked + non-ignored untracked 중
실제로 존재하는 authored 파일을 센다. 코드 LOC에는 주석·빈 줄·테스트가
포함되며, Markdown/JSON/lockfile은 LOC에서 제외한다.

| 항목 | Before | After | 감소 |
| --- | ---: | ---: | ---: |
| authored files | 180 | 21 | 88.3% |
| code files | 159 | 14 | 91.2% |
| code LOC | 25,897 | 730 | 97.2% |

남은 파일 수는 약 11.7%, 코드 줄 수는 약 2.8%다. 보고서, node_modules, CUDA
artifact, dist, 캐시와 외부 백업을 섞어 비교한 디스크 용량 수치는 아니다.

### 의도적으로 제거한 기능과 다음 단계

CUDA kernel/benchmark/수치 정책 실험, SASS/PC dataflow/profiler 연동,
operator knowledge index, 관계식 DSL/validator/projection, Python eager tracing,
범용 trace/plan/evidence value object, TS rewrite/capability engine,
Add 재결합 및 나머지 변환, graph 편집/JSON 편집기/LaTeX/복잡한 Explorer,
Kernel/Runtime/Hardware workspace를 제거했다. Neg/Abs/Transpose 등의
핵심 경로 밖 operator와 일반 shape/broadcast 확장도 유지하지 않았다.

다음 단계는 필요할 때 `cuda.py`의 선택된 구현 **하나**에 실제 launch와 출력
회수를 넣고 reference 결과와 비교하는 것이다. 그 실행에서 요구되는 dtype,
buffer 및 stream 계약만 추가한다. 실제 관측이 생기기 전에는 profiler UI나
새 catalog, 범용 rule framework를 복구하지 않는다.

**YES — 처음 보는 개발자는 `demo.py`와 핵심 Python 모듈 네 개를 순서대로
읽으면 Model → Graph → Operator Meaning → Backend Transformation → CUDA
interface를 추적할 수 있다.** Web은 이 결과의 소비자이며 별도 의미 체계가 아니다.
