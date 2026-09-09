"""Public web/news discovery for Yemen NewsGuard.

The engine is intentionally evidence-oriented:
- aggregators are discovery layers, not automatically original sources;
- source attribution is validated when possible;
- search uses multiple event angles instead of one keyword/site;
- results are scored for the same event, not merely broad topic overlap;
- public indexed content only. No login/private-content bypass is attempted.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; YemenNewsGuard/1.2; +https://github.com/Muoz22/yemen-newsguard-ai)"
}
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

AGGREGATOR_HOSTS = {
    "sahaafa.net": "صحافة نت",
}

STOPWORDS = {
    "من", "في", "على", "هذا", "هذه", "التي", "الذي", "عن", "إلى", "وقد", "كما",
    "أن", "إن", "تم", "مع", "بعد", "قبل", "الى", "هو", "هي", "كان", "كانت", "يتم",
    "لدى", "حتى", "بين", "ضمن", "حول", "ذلك", "تلك", "هناك", "اليمن", "اليمني",
}

@dataclass
class SearchResult:
    title: str
    url: str
    source: str
    published: str
    snippet: str
    match: float = 0.0
    channel: str = "ويب عام"
    searched_query: str = ""
    match_type: str = ""
    publisher_name: str = ""
    publisher_url: str = ""
    aggregator: str = ""
    attribution_status: str = "غير متحقق"
    relation: str = ""
    evidence_score: float = 0.0


def _request(url: str, timeout: int = 15) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response.text


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower().split(":")[0].removeprefix("www.")


def _canonical_url(url: str) -> str:
    try:
        parts = urlsplit(str(url).strip())
        if not parts.scheme or not parts.netloc:
            return str(url).strip()
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", "", ""))
    except Exception:
        return str(url).strip()


def _normalise(value: str) -> str:
    value = str(value or "").lower()
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    value = re.sub(r"[ـًٌٍَُِّْٱ]", "", value)
    value = re.sub(r"[^\wء-ي ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(value: str) -> list[str]:
    return [w for w in _normalise(value).split() if len(w) > 2 and w not in STOPWORDS]


def _token_set(value: str) -> set[str]:
    return set(_tokens(value))


def _domain_label(url: str) -> str:
    host = _host(url)
    return AGGREGATOR_HOSTS.get(host, host)


def _social_channel(url: str) -> str:
    host = _host(url)
    if host in {"facebook.com", "m.facebook.com", "fb.watch"} or host.endswith("facebook.com"):
        return "Facebook عام"
    if host in {"x.com", "twitter.com"} or host.endswith("x.com") or host.endswith("twitter.com"):
        return "X / Twitter عام"
    return ""


def _rss_results(url: str, channel: str) -> list[SearchResult]:
    try:
        soup = BeautifulSoup(_request(url), "html.parser")
    except requests.RequestException:
        return []
    results: list[SearchResult] = []
    for item in soup.find_all("item"):
        def text(tag_name: str) -> str:
            tag = item.find(tag_name)
            return tag.get_text(" ", strip=True) if tag else ""
        title = text("title")
        link = text("link")
        description = text("description")
        date = text("pubDate")
        source_node = item.find("source")
        source = source_node.get_text(" ", strip=True) if source_node else _domain_label(link)
        if title and link:
            results.append(SearchResult(
                title=BeautifulSoup(title, "html.parser").get_text(" ", strip=True),
                url=link,
                source=source,
                published=date or "",
                snippet=BeautifulSoup(description or "", "html.parser").get_text(" ", strip=True)[:1000],
                channel=channel,
            ))
    return results


def _bing_html_results(query: str, channel: str = "Bing Web") -> list[SearchResult]:
    try:
        soup = BeautifulSoup(_request("https://www.bing.com/search?q=" + quote(query[:240])), "html.parser")
    except requests.RequestException:
        return []
    results: list[SearchResult] = []
    for card in soup.select("li.b_algo")[:12]:
        anchor = card.select_one("h2 a")
        if not anchor or not anchor.get("href"):
            continue
        url = anchor["href"]
        snippet_node = card.select_one(".b_caption p")
        results.append(SearchResult(
            title=anchor.get_text(" ", strip=True),
            url=url,
            source=_domain_label(url),
            published="",
            snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
            channel=channel,
            searched_query=query,
        ))
    return results


def _tavily_results(query: str) -> list[SearchResult]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.strip().lower() in {"مفتاح tavily الخاص بك", "your tavily key", "ضع-المفتاح-هنا"}:
        return []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": 10,
                "include_answer": False,
                "include_raw_content": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        items = response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError):
        return []
    results: list[SearchResult] = []
    for item in items:
        url = str(item.get("url", ""))
        if not url:
            continue
        results.append(SearchResult(
            title=str(item.get("title", "")),
            url=url,
            source=_domain_label(url),
            published=str(item.get("published_date", "")),
            snippet=str(item.get("raw_content") or item.get("content") or "")[:12000],
            channel="فهرس ويب موسع",
            searched_query=query,
        ))
    return results


def _ai_search_plan(claim: str) -> list[str]:
    """Create event-oriented search angles without inventing facts."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    prompt = (
        "أنت مخطط بحث إخباري. حلل الادعاء التالي إلى 8 استعلامات بحث عربية مستقلة للعثور على "
        "تغطيات نفس الحدث، وليس أخباراً عامة عن الموضوع. استخرج فقط الأسماء والأماكن والأفعال "
        "والعبارات الموجودة في الادعاء، ولا تخترع أي معلومة. نوّع الزوايا: العبارة المميزة، "
        "الأشخاص/الجهات، المكان+الفعل، الفعل+الشخص، النتائج/التبعات المذكورة، وصياغة بديلة. "
        "لا تستخدم site:sahaafa.net. أعد JSON فقط بالمفتاح queries.\nالادعاء: " + claim
    )
    try:
        response = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("NEWSGUARD_SEARCH_MODEL", "gpt-4o-mini"),
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = json.loads(response.json()["choices"][0]["message"]["content"])
        return list(dict.fromkeys(str(q).strip() for q in data.get("queries", []) if str(q).strip()))[:8]
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return []


def _fallback_search_plan(claim: str) -> list[str]:
    words = _tokens(claim)
    variants = [claim[:200]]
    if len(words) >= 4:
        variants += [
            " ".join(words[:8]),
            " ".join(words[-8:]),
            " ".join(words[:4]),
            " ".join(words[-5:]),
        ]
    return list(dict.fromkeys(v for v in variants if v.strip()))[:6]


def _extract_sahaafa_attribution(soup: BeautifulSoup, page_url: str) -> tuple[str, str]:
    """Extract the publisher name AND actual href from Sahaafa without guessing."""
    candidates: list[tuple[str, str]] = []
    for text_node in soup.find_all(string=re.compile(r"المصدر\s*:?") if soup else None):
        parent = getattr(text_node, "parent", None)
        if not parent:
            continue
        # Prefer links in the same small container as the source marker.
        containers = [parent]
        if parent.parent:
            containers.append(parent.parent)
        for container in containers:
            for link in container.find_all("a", href=True):
                name = link.get_text(" ", strip=True)
                href = urljoin(page_url, link.get("href", ""))
                if name and href and "صحافة نت" not in name and _host(href) != "sahaafa.net":
                    candidates.append((name, href))
            if candidates:
                break
    if candidates:
        return candidates[0]

    # Metadata is a fallback only; it is not treated as proof of an original publisher URL.
    for meta in soup.find_all("meta"):
        prop = str(meta.get("property") or meta.get("name") or "").lower()
        content = str(meta.get("content") or "").strip()
        if prop in {"article:publisher", "og:site_name"} and content and content != "صحافة نت":
            return content, ""
    return "", ""


def _jsonld_publisher(soup: BeautifulSoup) -> str:
    for node in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        try:
            data = json.loads(node.string or node.get_text())
        except (ValueError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            publisher = item.get("publisher")
            if isinstance(publisher, dict):
                name = publisher.get("name")
                if name:
                    return str(name).strip()
            elif isinstance(publisher, str) and publisher.strip():
                return publisher.strip()
    return ""


def _validate_attribution(result: SearchResult, soup: BeautifulSoup, html: str) -> SearchResult:
    host = _host(result.url)
    if host not in AGGREGATOR_HOSTS:
        result.publisher_name = result.source if result.source and result.source != host else ""
        result.publisher_url = result.url
        result.aggregator = ""
        result.attribution_status = "مباشر"
        return result

    result.aggregator = AGGREGATOR_HOSTS[host]
    name, href = _extract_sahaafa_attribution(soup, result.url)
    if name:
        result.publisher_name = name
    if href:
        result.publisher_url = _canonical_url(href)

    # The aggregator itself is always the observed URL. Never label it as original.
    result.source = result.publisher_name or "غير متحقق"
    result.attribution_status = "منسوب فقط"

    if result.publisher_url and _host(result.publisher_url) != host:
        try:
            original_html = _request(result.publisher_url, timeout=10)
            original_soup = BeautifulSoup(original_html, "html.parser")
            original_title = ""
            og = original_soup.find("meta", attrs={"property": "og:title"})
            if og and og.get("content"):
                original_title = og.get("content", "")
            if not original_title:
                h1 = original_soup.find("h1")
                original_title = h1.get_text(" ", strip=True) if h1 else (original_soup.title.get_text(" ", strip=True) if original_soup.title else "")
            aggregator_title = result.title
            title_score = SequenceMatcher(None, _normalise(aggregator_title), _normalise(original_title)).ratio()
            source_text = original_soup.get_text(" ", strip=True)[:16000]
            event_overlap = len(_token_set(aggregator_title) & _token_set(source_text)) / max(1, len(_token_set(aggregator_title)))
            if title_score >= 0.72 or event_overlap >= 0.60:
                result.attribution_status = "متحقق"
                result.source = result.publisher_name or _jsonld_publisher(original_soup) or _host(result.publisher_url)
            else:
                result.attribution_status = "غير متحقق"
        except requests.RequestException:
            result.attribution_status = "غير متحقق"
    elif result.publisher_name:
        result.attribution_status = "منسوب فقط"
    else:
        result.attribution_status = "غير متحقق"
        result.source = "غير متحقق"
    return result


def _enrich_page(result: SearchResult) -> SearchResult:
    try:
        html = _request(result.url, timeout=12)
        soup = BeautifulSoup(html, "html.parser")
        if "sahaafa.net" in _host(result.url):
            result = _validate_attribution(result, soup, html)
        else:
            publisher = _jsonld_publisher(soup)
            if not result.source or result.source == _host(result.url):
                result.source = publisher or _domain_label(result.url)
            result.publisher_name = result.source
            result.publisher_url = _canonical_url(result.url)
            result.attribution_status = "مباشر"
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        body = soup.get_text(" ", strip=True)
        if body:
            result.snippet = f"{result.snippet} {body[:14000]}"[:16000]
        # Prefer explicit article publication metadata when search result has none.
        if not result.published:
            for meta_name in ["article:published_time", "datePublished", "pubdate"]:
                meta = soup.find("meta", attrs={"property": meta_name}) or soup.find("meta", attrs={"name": meta_name})
                if meta and meta.get("content"):
                    result.published = str(meta.get("content"))
                    break
    except requests.RequestException:
        pass
    return result


def _user_provided_results(query: str) -> tuple[str, list[SearchResult]]:
    urls = re.findall(r"https?://[^\s<>]+", query or "")
    clean_query = re.sub(r"https?://[^\s<>]+", " ", query or "")
    results: list[SearchResult] = []
    for raw_url in dict.fromkeys(urls):
        url = raw_url.rstrip(".,،؛")
        host = _host(url)
        channel = _social_channel(url) or "رابط مقدم من المستخدم"
        result = SearchResult(
            title="رابط أدخله المستخدم", url=url, source=host, published="", snippet="",
            match=0.0, channel=channel, searched_query=clean_query.strip(),
            match_type="رابط مباشر مقدم من المستخدم",
        )
        result = _enrich_page(result)
        if result.title == "رابط أدخله المستخدم":
            result.title = host
        results.append(result)
    return clean_query.strip(), results


def _event_components(claim: str) -> dict[str, set[str]]:
    tokens = _token_set(claim)
    # Lightweight extraction that works without an LLM. The AI planner supplies richer angles when enabled.
    locations = {w for w in tokens if w in {"صنعاء", "عدن", "تعز", "الضالع", "مارب", "مأرب", "الحديدة", "الجوف", "شبوة", "ابين", "أبين", "حضرموت", "البيضاء", "الزاهر", "المخا"}}
    actions = {w for w in tokens if w in {"استهداف", "استهدف", "اشتباكات", "اشتباك", "قصف", "انفجار", "مقتل", "قتل", "إصابة", "اصابة", "وصول", "زيارة", "صرف", "إحراق", "احراق", "تقدم", "تعزيزات", "اعتراض", "إسقاط", "اسقاط"}}
    return {"tokens": tokens, "locations": locations, "actions": actions}


def _score_event(claim: str, result: SearchResult) -> float:
    claim_tokens = _token_set(claim)
    page_text = f"{result.title} {result.snippet}"
    page_tokens = _token_set(page_text)
    if not claim_tokens or not page_tokens:
        return 0.0
    overlap = len(claim_tokens & page_tokens) / max(1, len(claim_tokens))
    title_sim = SequenceMatcher(None, _normalise(claim), _normalise(result.title)).ratio()
    body_sim = 0.0
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), max_features=6000).fit_transform([claim, page_text[:12000]])
        body_sim = float(cosine_similarity(matrix[0:1], matrix[1:2]).ravel()[0])
    except ValueError:
        pass
    components = _event_components(claim)
    location_overlap = len(components["locations"] & page_tokens) / max(1, len(components["locations"])) if components["locations"] else 0.0
    action_overlap = len(components["actions"] & page_tokens) / max(1, len(components["actions"])) if components["actions"] else 0.0
    phrase_bonus = 0.0
    for size in (8, 7, 6, 5):
        q = _tokens(claim)
        for start in range(max(0, len(q) - size + 1)):
            phrase = " ".join(q[start:start + size])
            if phrase and phrase in _normalise(page_text):
                phrase_bonus = 0.20
                break
        if phrase_bonus:
            break
    score = (
        overlap * 0.30 +
        title_sim * 0.25 +
        body_sim * 0.15 +
        location_overlap * 0.15 +
        action_overlap * 0.10 +
        phrase_bonus
    )
    # Broad topical overlap must not become a false 100% match.
    if overlap < 0.45 and title_sim < 0.70:
        score *= 0.55
    return round(min(0.99, score), 2)


def _relation(score: float) -> str:
    if score >= 0.78:
        return "نفس الحدث"
    if score >= 0.55:
        return "سياق مرتبط"
    return "غير ذي صلة"


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    unique: dict[str, SearchResult] = {}
    for result in results:
        key = _canonical_url(result.url)
        if not key:
            continue
        if key not in unique:
            unique[key] = result
            continue
        old = unique[key]
        # Keep the richer observation, while preserving a verified attribution.
        if result.attribution_status == "متحقق" and old.attribution_status != "متحقق":
            unique[key] = result
        elif len(result.snippet) > len(old.snippet):
            unique[key] = result
    return list(unique.values())


def _diversify(results: list[SearchResult], limit: int) -> list[SearchResult]:
    """Prevent one host/aggregator from consuming the entire result set."""
    chosen: list[SearchResult] = []
    host_counts: dict[str, int] = {}
    max_per_host = max(3, min(5, limit // 5 if limit >= 15 else 3))
    for result in results:
        host = _host(result.publisher_url or result.url)
        if host in AGGREGATOR_HOSTS:
            # Aggregators get a tighter cap; they are discovery evidence, not independent sources.
            cap = 3
        else:
            cap = max_per_host
        if host_counts.get(host, 0) >= cap:
            continue
        chosen.append(result)
        host_counts[host] = host_counts.get(host, 0) + 1
        if len(chosen) >= limit:
            break
    return chosen


def search_public_web(query: str, sources_path: Path, max_results: int = 25) -> list[dict]:
    query, direct_results = _user_provided_results(query)
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        return [asdict(item) for item in direct_results]

    ai_variants = _ai_search_plan(query)
    variants = list(dict.fromkeys(ai_variants + _fallback_search_plan(query)))[:10]
    collected: list[SearchResult] = list(direct_results)

    # Multiple independent search surfaces. No Sahaafa-only search is used.
    for search_query in variants:
        collected.extend(_tavily_results(search_query))
        encoded = quote(search_query[:240])
        feeds = [
            (f"https://news.google.com/rss/search?q={encoded}&hl=ar&gl=YE&ceid=YE:ar", "Google News"),
            (f"https://www.bing.com/news/search?q={encoded}&format=rss", "Bing News"),
        ]
        for url, channel in feeds:
            for result in _rss_results(url, channel):
                result.searched_query = search_query
                collected.append(result)
        collected.extend(_bing_html_results(search_query))

    # Configured sources are supplementary, never the dominant search path.
    if sources_path.exists():
        try:
            sources = pd.read_csv(sources_path).fillna("")
            for _, source in sources.iterrows():
                if str(source.get("active", "TRUE")).upper() != "TRUE":
                    continue
                base_url = str(source.get("url", "")).rstrip("/")
                if not base_url.startswith("http") or _host(base_url) in AGGREGATOR_HOSTS:
                    continue
                # Only fetch the configured homepage here; search engines remain primary discovery.
                try:
                    soup = BeautifulSoup(_request(base_url, timeout=10), "html.parser")
                except requests.RequestException:
                    continue
                for link in soup.find_all("a", href=True)[:150]:
                    title = link.get_text(" ", strip=True)
                    if len(title) < 15 or len(title) > 300:
                        continue
                    url = urljoin(base_url, link["href"])
                    collected.append(SearchResult(
                        title=title, url=url,
                        source=str(source.get("name", _host(url))), published="",
                        snippet="نتيجة من مصدر يمني مُهيأ", channel="مصدر يمني مباشر",
                        searched_query=query,
                    ))

    collected = _deduplicate(collected)

    # Enrich only the best broad candidate pool to control latency.
    preliminary: list[SearchResult] = []
    for result in collected:
        preliminary.append(result)
    preliminary.sort(key=lambda r: len(r.snippet), reverse=True)
    candidates = preliminary[:100]
    for index, item in enumerate(candidates):
        candidates[index] = _enrich_page(item)

    for item in candidates:
        item.match = _score_event(query, item)
        item.relation = _relation(item.match)
        item.evidence_score = item.match
        social = _social_channel(item.url)
        if social:
            item.channel = social
        if item.attribution_status == "متحقق":
            item.match_type = "مصدر أصلي متحقق — نفس الحدث" if item.match >= 0.78 else "مصدر أصلي متحقق — سياق مرتبط"
        elif item.aggregator:
            item.match_type = "مجمع/فهرسة — مصدر منسوب" if item.match >= 0.55 else "مجمع/فهرسة — صلة ضعيفة"
        elif social:
            item.match_type = "منصة اجتماعية عامة — نفس الحدث" if item.match >= 0.78 else "منصة اجتماعية عامة — سياق مرتبط"
        elif item.match >= 0.78:
            item.match_type = "تطابق قوي"
        elif item.match >= 0.55:
            item.match_type = "سياق مرتبط"
        else:
            item.match_type = "غير ذي صلة"

    # Keep useful evidence only. Social/public links can survive at a lower threshold for later stages.
    ranked = [
        item for item in candidates
        if item.match >= 0.50 or (_social_channel(item.url) and item.match >= 0.35)
    ]
    ranked.sort(key=lambda r: (r.match, r.attribution_status == "متحقق", bool(r.publisher_url)), reverse=True)
    ranked = _diversify(ranked, max_results)
    return [asdict(item) for item in ranked]
