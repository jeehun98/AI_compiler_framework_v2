# AICF Graph Lab

연산자를 Scratch처럼 조합하고 계산 그래프의 수학적 의미, 검증 결과, 적용 가능한 rewrite를 탐색하는 로컬 React 애플리케이션입니다. 수학적 의미 그래프는 실행 backend와 분리되어 있으며 Python/CUDA 실행은 포함하지 않습니다.

## 설치와 실행

Node.js가 설치된 환경에서 이 디렉터리로 이동해 실행합니다.

```powershell
npm install
npm run dev
```

Vite가 출력한 로컬 주소(기본값 `http://localhost:5173`)를 브라우저에서 엽니다.

## 사용 흐름

1. 왼쪽 팔레트의 노드를 클릭하거나 캔버스로 드래그합니다.
2. 노드 오른쪽 output handle에서 대상 노드의 왼쪽 input handle로 연결합니다.
3. 헤더와 캔버스의 Validation 패널에서 필수 입력, 포트 중복, dangling edge, cycle, output 오류를 확인합니다.
4. 아래 수식 영역에서 명시적 graph output별 KaTeX 수식을 확인합니다.
5. 노드를 선택해 의미, arity, shape 규칙, 대수적 성질, 수치적 주의점, 적용 규칙, 네 자유도 축을 확인합니다.
6. rewrite 후보를 선택해 원본과 후보를 나란히 비교하고, `이 후보 적용`을 눌러 명시적으로 반영합니다.
7. `저장`으로 JSON을 다운로드하고 `JSON 불러오기`로 다시 불러옵니다. 잘못된 문서는 현재 그래프를 변경하지 않습니다.

제공 예제는 `x × 1`, `(2 × 3) + 0`, `Transpose(Transpose(x))`입니다.

## 지원 연산자

- Input
- Constant
- Add
- Mul
- MatMul
- ReLU
- Transpose
- ReduceSum

## 지원 rewrite 규칙

| 규칙 | 분류 | 주요 조건 |
| --- | --- | --- |
| `x + 0 → x` | conditionally exact | signed zero 관찰 가능성 확인 |
| `x × 1 → x` | conditionally exact | NaN signaling/payload 정책 확인 |
| `x × 0 → 0` | conditionally exact | NaN, Inf, signed zero 정책 및 유한 입력 조건 확인 |
| constant folding | conditionally exact | JavaScript number와 target dtype 의미 일치 |
| Add 입력 교환 | conditionally exact | NaN payload와 평가 순서가 비관찰적이어야 함 |
| Mul 입력 교환 | conditionally exact | NaN payload와 평가 순서가 비관찰적이어야 함 |
| 연속 Transpose 제거 | exact | 두 연산 모두 마지막 두 축 교환 의미 |

Rewrite는 입력 graph를 변경하지 않는 새 graph를 만들고, 적용 전후 validation을 통과한 후보만 노출합니다. `x × 0 → 0` 같은 의미 rewrite는 사용되지 않는 upstream 노드를 암묵적으로 삭제하지 않습니다. 필요 시 `cleanupUnusedNodes`를 별도 단계로 호출할 수 있습니다.

## 테스트와 빌드

```powershell
npm run typecheck
npm test -- src/core
npm test -- src/store/graphStore.test.ts src/App.test.tsx
npm test
npm run build
```

Vitest는 Windows의 메모리 사용과 worker 생성 부담을 줄이기 위해 `fileParallelism: false`, `maxWorkers: 1`로 설정되어 있습니다.

## 구조

```text
src/
├─ catalog/       # Operator 의미·shape·자유도 metadata
├─ core/          # validation, expression, rewrite, codec, topology, cleanup
├─ domain/        # Operator, Graph, RewriteRule, ValidationResult 타입
├─ examples/      # 완전한 GraphDocument 예제
├─ store/         # Zustand 단일 문서 상태와 편집 action
├─ backend/       # 미래 외부 backend adapter 인터페이스만 정의
├─ components/    # React Flow custom node
└─ App.tsx        # UI 조합과 React Flow 이벤트 wiring
```

`Graph.outputs`는 수식과 rewrite가 공유하는 명시적 결과 계약입니다. 다중 output은 배열 순서를 보존해 각각 `y₁`, `y₂`로 표시합니다. JSON codec은 operator별 parameters, ID 중복, 포트, 참조 node, layout, outputs를 런타임에서 검증합니다.

## 현재 한계

- 일반적인 shape inference나 broadcast/axis의 정적 증명은 하지 않습니다.
- 수치 실행, dtype 선택, autograd, code generation, Python/CUDA backend는 구현하지 않습니다.
- 교환법칙 후보는 fingerprint와 현재 세션의 방문 상태로 왕복 반복을 완화하지만 범용 canonicalization은 하지 않습니다.
- dead-code cleanup은 rewrite와 분리된 명시적 core 함수이며 UI action으로는 아직 노출하지 않습니다.
- JSON schema version은 현재 `1`만 지원합니다.
