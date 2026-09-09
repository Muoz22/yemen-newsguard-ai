"""Structured monitoring and reporting layer for Yemen NewsGuard AI."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .web_search import search_public_web


def _platform(url: str, channel: str = "") -> str:
    host = urlsplit(url).netloc.lower()
    if "facebook.com" in host or "fb.watch" in host:
        return "Facebook"
    if "x.com" in host or "twitter.com" in host:
        return "X / Twitter"
    if "youtube.com" in host or "youtu.be" in host:
        return "YouTube"
    if "t.me" in host or "telegram.me" in host:
        return "Telegram"
    if "tiktok.com" in host:
        return "TikTok"
    if "instagram.com" in host:
        return "Instagram"
    if channel:
        if "Facebook" in channel: return "Facebook"
        if "X /" in channel or "Twitter" in channel: return "X / Twitter"
    return "موقع ويب"


def _account(url: str) -> str:
    parts = [p for p in urlsplit(url).path.split("/") if p]
    host = urlsplit(url).netloc.lower()
    if not parts:
        return ""
    if "facebook.com" in host:
        if parts[0] in {"share", "sharer", "watch", "reel", "plugins"}:
            return ""
        return parts[0]
    if "x.com" in host or "twitter.com" in host:
        return parts[0]
    if "t.me" in host or "telegram.me" in host:
        return parts[0]
    return ""


def _domain(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def monitor_claim(claim: str, sources_path: Path, max_results: int = 50) -> dict:
    rows = search_public_web(claim, sources_path, max_results=max_results)
    evidence = []
    seen = set()
    for row in rows:
        url = str(row.get("url", ""))
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        if not url or key in seen:
            continue
        seen.add(key)
        platform = _platform(url, str(row.get("channel", "")))
        evidence.append({
            "id": key,
            "title": str(row.get("title", "")),
            "url": url,
            "domain": _domain(url),
            "platform": platform,
            "account_or_page": _account(url),
            "source": str(row.get("source", "")),
            "published": str(row.get("published", "")),
            "match": float(row.get("match", 0) or 0),
            "match_type": str(row.get("match_type", "")),
            "query": str(row.get("searched_query", "")),
            "snippet": str(row.get("snippet", ""))[:1200],
        })
    evidence.sort(key=lambda x: x["match"], reverse=True)
    strong = [x for x in evidence if x["match"] >= 0.75]
    domains = Counter(x["domain"] for x in evidence if x["domain"])
    platforms = Counter(x["platform"] for x in evidence)
    accounts = sorted({x["account_or_page"] for x in evidence if x["account_or_page"]})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim": claim,
        "total_results": len(evidence),
        "strong_matches": len(strong),
        "domains": dict(domains),
        "platforms": dict(platforms),
        "accounts_or_pages": accounts,
        "evidence": evidence,
    }


def report_csv(report: dict) -> bytes:
    output = io.StringIO()
    fields = ["id", "title", "url", "domain", "platform", "account_or_page", "source", "published", "match", "match_type", "query", "snippet"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report.get("evidence", []))
    return output.getvalue().encode("utf-8-sig")


def report_json(report: dict) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")


def report_html(report: dict) -> bytes:
    rows = []
    for x in report.get("evidence", []):
        rows.append(
            f"<tr><td>{x['platform']}</td><td>{x['domain']}</td><td>{x['account_or_page']}</td>"
            f"<td>{x['match']:.0%}</td><td>{x['match_type']}</td><td><a href='{x['url']}'>فتح</a></td>"
            f"<td>{x['title']}</td></tr>"
        )
    html = f"""<!doctype html><html lang='ar' dir='rtl'><meta charset='utf-8'><title>تقرير رصد الخبر</title>
<style>body{{font-family:Arial,sans-serif;margin:30px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:right}}th{{background:#eee}}.note{{padding:12px;background:#fff6d8}}</style>
<h1>تقرير رصد الخبر</h1><p><b>الخبر/الادعاء:</b> {report['claim']}</p><p><b>وقت إنشاء التقرير:</b> {report['generated_at']}</p>
<p><b>النتائج:</b> {report['total_results']} — <b>التطابقات القوية:</b> {report['strong_matches']}</p>
<div class='note'>هذا التقرير يرصد نتائج الويب العامة المفهرسة والمتاحة فقط. عدم ظهور نتيجة لا يعني عدم وجودها، ولا يعني التطابق وحده أن الخبر صحيح.</div>
<table><tr><th>المنصة</th><th>النطاق</th><th>الحساب/الصفحة</th><th>التطابق</th><th>نوع التطابق</th><th>الرابط</th><th>العنوان</th></tr>{''.join(rows)}</table></html>"""
    return html.encode("utf-8")
