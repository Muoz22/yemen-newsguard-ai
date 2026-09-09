"""News fingerprinting, historical observations, and propagation intelligence."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid", "mc_cid", "mc_eid"}


def normalize_arabic(text: str) -> str:
    text = str(text or "").lower()
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[ـ]+", "", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_url(url: str) -> str:
    try:
        p = urlsplit(url.strip())
        query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMS]
        path = p.path.rstrip("/") or "/"
        return urlunsplit((p.scheme.lower(), p.netloc.lower().removeprefix("www."), path, urlencode(query), ""))
    except Exception:
        return url.strip()


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_arabic(text).encode("utf-8")).hexdigest()


def key_phrases(text: str, limit: int = 5) -> list[str]:
    words = normalize_arabic(text).split()
    phrases: list[str] = []
    if len(words) >= 5:
        phrases.extend([" ".join(words[:8]), " ".join(words[-8:])])
    for i in range(0, min(max(len(words) - 3, 0), 12), 3):
        phrases.append(" ".join(words[i:i + 6]))
    return list(dict.fromkeys(p for p in phrases if p))[:limit]


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS stories (
            story_id TEXT PRIMARY KEY, claim TEXT NOT NULL, normalized_claim TEXT NOT NULL,
            fingerprint TEXT NOT NULL, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS observations (
            observation_id TEXT PRIMARY KEY, story_id TEXT NOT NULL, url TEXT NOT NULL,
            canonical_url TEXT NOT NULL, title TEXT, domain TEXT, platform TEXT,
            account_or_page TEXT, source TEXT, published TEXT, discovered_at TEXT NOT NULL,
            first_seen TEXT, last_seen TEXT, match REAL, match_type TEXT, snippet TEXT,
            search_query TEXT, UNIQUE(story_id, canonical_url)
        );
        CREATE INDEX IF NOT EXISTS idx_obs_story ON observations(story_id);
        CREATE INDEX IF NOT EXISTS idx_obs_domain ON observations(domain);
        CREATE INDEX IF NOT EXISTS idx_obs_platform ON observations(platform);
        CREATE INDEX IF NOT EXISTS idx_obs_discovered ON observations(discovered_at);
        """)
        cols = {row[1] for row in con.execute("PRAGMA table_info(observations)")}
        if "first_seen" not in cols:
            con.execute("ALTER TABLE observations ADD COLUMN first_seen TEXT")
        if "last_seen" not in cols:
            con.execute("ALTER TABLE observations ADD COLUMN last_seen TEXT")


def story_id(claim: str) -> str:
    return fingerprint(claim)[:24]


def persist_report(report: dict, db_path: Path) -> dict:
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    sid = story_id(report.get("claim", ""))
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO stories(story_id,claim,normalized_claim,fingerprint,first_seen,last_seen) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(story_id) DO UPDATE SET last_seen=excluded.last_seen",
            (sid, report.get("claim", ""), normalize_arabic(report.get("claim", "")), fingerprint(report.get("claim", "")), now, now),
        )
        for item in report.get("evidence", []):
            url = str(item.get("url", ""))
            if not url:
                continue
            cu = canonical_url(url)
            oid = hashlib.sha256(f"{sid}|{cu}".encode("utf-8")).hexdigest()[:32]
            con.execute(
                """INSERT INTO observations(
                    observation_id,story_id,url,canonical_url,title,domain,platform,account_or_page,
                    source,published,discovered_at,first_seen,last_seen,match,match_type,snippet,search_query
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(story_id,canonical_url) DO UPDATE SET
                    url=excluded.url, title=excluded.title, published=excluded.published,
                    last_seen=excluded.last_seen, match=MAX(observations.match, excluded.match),
                    match_type=excluded.match_type, snippet=excluded.snippet, search_query=excluded.search_query
                """,
                (oid, sid, url, cu, item.get("title", ""), item.get("domain", ""), item.get("platform", ""),
                 item.get("account_or_page", ""), item.get("source", ""), item.get("published", ""), now, now, now,
                 float(item.get("match", 0) or 0), item.get("match_type", ""), item.get("snippet", ""), item.get("query", "")),
            )
    return propagation_summary(report, sid, db_path)


def propagation_summary(report: dict, sid: str, db_path: Path) -> dict:
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM observations WHERE story_id=? ORDER BY last_seen DESC", (sid,)
        ).fetchall()]
    evidence = report.get("evidence", [])
    domains = Counter(x.get("domain") for x in rows if x.get("domain"))
    platforms = Counter(x.get("platform") for x in rows if x.get("platform"))
    strong = [x for x in rows if float(x.get("match", 0) or 0) >= 0.75]
    # Only call a timestamp a candidate when it comes from the source metadata.
    dated = [x for x in rows if x.get("published") and str(x.get("published")).strip()]
    dated.sort(key=lambda x: str(x.get("published")))
    return {
        "story_id": sid,
        "first_observed_candidate": dated[0] if dated else (rows[-1] if rows else None),
        "observations": len(rows),
        "strong_matches": len(strong),
        "domain_counts": dict(domains),
        "platform_counts": dict(platforms),
        "history": rows,
        "current_evidence": evidence,
    }
