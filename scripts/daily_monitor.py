from pathlib import Path
import json
import os
from datetime import datetime, timezone

import pandas as pd

from src.monitoring import monitor_claim

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "data" / "monitor_queries.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

queries = pd.read_csv(QUERIES).fillna("")
all_reports = []
for _, row in queries.iterrows():
    if str(row.get("active", "TRUE")).upper() != "TRUE":
        continue
    query = str(row.get("query", "")).strip()
    if not query:
        continue
    report = monitor_claim(query, ROOT / "data" / "sources.csv", max_results=50)
    report["query_id"] = str(row.get("query_id", ""))
    report["query_name"] = str(row.get("name", ""))
    all_reports.append(report)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
path = OUT / f"daily_{stamp}.json"
path.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Created {path} with {len(all_reports)} monitored queries")
