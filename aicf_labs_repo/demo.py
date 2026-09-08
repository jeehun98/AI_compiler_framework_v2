"""One complete slice. Run: python demo.py --output web/public/report.json"""

import argparse
import json
from pathlib import Path
from aicf import Linear, ReLU, Sequential, run


def tiny_model():
    return Sequential(
        Linear(weight=[[1, -1, 2], [0, 3, -1]], bias=[1, 0, -2]),
        ReLU(),
        Linear(weight=[[2], [-1], [3]], bias=[4]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(tiny_model(), [[1, 2], [3, 4]])
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"Saved {args.output}; reference outputs equal: {report['execution']['equal']}")
    else:
        print(encoded)
