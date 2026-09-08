# AICF — 최소 모델 변환 골격

**읽는 순서: `model.py` → `operators.py` → `backend.py` → `cuda.py`.**

```text
Sequential(Linear, ReLU, Linear)
  → MatMul → Add → ReLU → MatMul → Add
  → LinearReLU → MatMul → Add
  → reference 결과 비교
  → CUDA implementation ID 선택
  → launch interface: not-implemented
```

## 실행

Python 3.10 이상과 웹 개발용 Node.js/npm이 필요하다. Python 외부 패키지는 없다.
이 README가 있는 `aicf_labs_repo`에서 실행한다.

```powershell
python demo.py
python demo.py --output web/public/report.json
python -m unittest discover -s tests -v
```

예제 입력은 `[[1, 2], [3, 4]]`, 원본/변환 결과는 모두 `[[3], [3]]`다.

```python
from aicf import Linear, ReLU, Sequential, run

model = Sequential(
    Linear([[1, -1, 2], [0, 3, -1]], [1, 0, -2]),
    ReLU(),
    Linear([[2], [-1], [3]], [4]),
)
report = run(model, [[1, 2], [3, 4]])
assert report["execution"]["equal"]
```

## 구조

```text
aicf/
  model.py       사용자 layer가 operator와 graph를 생성
  operators.py   operator instance, shape 계약, 유일한 의미/property 정의
  backend.py     두 rule, legality, reference 실행, implementation 선택
  cuda.py        implementation lookup과 미구현 launch interface
demo.py          위 경로를 실행하고 JSON report 생성
tests/           핵심 Python 테스트
web/src/         Python report를 표시하는 React viewer
```

Operator 의미는 `operators.py`에만 정의한다. Web에는 catalog나 rule이 없다.
JSON은 Python 실행 결과이며 직접 편집하는 의미 정보가 아니다.
`input`, `constant` 경계 노드 외에 MatMul/Add/Mul/ReLU/ReduceSum/LinearReLU만
남겼다. shape는 구체 크기이며 scalar·동일 shape·matrix/vector broadcast,
2D MatMul과 all-axis ReduceSum 범위만 지원한다.

## Backend

`transform(graph)`는 기본적으로 `linear_relu`를 적용한다.
`transform(graph, rule="positive_scale")`로 두 번째 rule을 선택한다.

- `linear_relu`: private MatMul → constant [N] bias Add → ReLU를 결합한다.
- `positive_scale`: 입력 0의 비음수 동차성에 따라 scalar를 operator 뒤로 옮긴다.

상태는 `ACCEPT / REJECT / UNKNOWN`이다. 선언되지 않은 의미는 UNKNOWN,
음수 scale과 공유 intermediate 등 명시적 제약 위반은 REJECT다.
규칙은 실수 수학을 기준으로 판단한다. `domain="ieee"`처럼 실행 의미를
요청하면 근거 없이 승인하지 않는다. Reference 비교 역시 해당 입력의 수치
비교이며 모든 입력이나 GPU의 bitwise 동치를 증명하지 않는다.

## CUDA 경계

`select_implementations(graph)`가 계산 노드마다 `cuda.lookup(type)`으로 ID를
선택한다. 예: `cuda.linear_relu.placeholder`. `run()`은 각 선택의 `launch()`까지
호출하지만 모든 CUDA 구현은 `not-implemented`, output은 null, observations는
빈 목록을 반환한다. 표시되는 실제 출력은 **reference backend의 출력**이다.
CUDA가 구현되면 `cuda.py`의 해당 구현부터 교체한다.

## 최소 Web

```powershell
cd web
npm install
npm run dev
npm test
npm run typecheck
npm run build
```

각 명령은 먼저 Python demo를 실행해 `public/report.json`을 생성한다.
화면은 모델, 원본/변환 graph의 노드와 edge, 선택 operator 의미, backend 결정,
reference 비교와 선택된 CUDA ID만 보여준다. `demo.py`를 수정했다면
`npm run report`로 보고서를 다시 생성한다. GPU 측정값이나 timeline은 없다.

기존 기능을 복구하려면 [REDUCTION.md](REDUCTION.md)의 Git 기준점과 작업 트리
밖 백업을 참고한다. 해당 문서에 삭제 전 inventory, 설계와 크기 변화도 기록했다.
