"""Build a propagation report from a saved JSON monitor report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.intelligence import persist_report

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/build_propagation_report.py reports/file.json")
    path = Path(sys.argv[1])
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = persist_report(report, ROOT / "data" / "newsguard_history.db")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
