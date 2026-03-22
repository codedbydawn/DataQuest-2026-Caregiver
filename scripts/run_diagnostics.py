from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damb.diagnostics import run_diagnostics


def main() -> None:
    results = run_diagnostics()
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
