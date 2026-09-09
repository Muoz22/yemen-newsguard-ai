"""Public web/news discovery for Yemen NewsGuard AI.

Evidence-oriented search: aggregators are discovery layers, source attribution is
validated when possible, and results are ranked by event relevance and diversity.
Only public/indexed content is used.
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

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; YemenNewsGuard/1.3)"}
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

AGGREGATOR_HOSTS = {"sahaafa.net": "صحافة نت"}
STOPWORDS = {"من", "في", "على", "هذا", "هذه", "التي", "الذي", "عن", "الى", "إلى", "وقد", "كما", "أن", "إن", "تم", "مع", "بعد", "قبل", "هو", "هي", "كان", "كانت", "اليمن", "اليمني"}
LOCATIONS = {"صنعاء", "عدن", "تعز", "الضالع", "مارب", "مأرب", "الحديدة", "الجوف", "شبوة", "ابين", "أبين", "حضرموت", "البيضاء", "الزاهر", "المخا"}
ACTIONS = {"استهداف", "استهدف", "اشتباكات", "اشتباك", "قصف", "انفجار", "مقتل", "قتل", "إصابة", "اصابة", "وصول", "زيارة", "صرف", "إحراق", "احراق", "تقدم", "تعزيزات", "اعتراض", "إسقاط", "اسقاط"}


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
    try:
        return urlsplit(str(url)).netloc.lower().split(":")[0].removeprefix("www.")
    except Exception:
        return ""


def _canonical_url(url: str) -> str:
    try:
        parts = urlsplit(str(url).strip())
        if not parts.scheme or not parts.netloc:
            return str(url).strip()
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", "", ""))
    except Exception:
        return str(url).strip()


def _normalise(value: str) -> str:
    text = str(value or "").lower()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    text = re.sub(r"[ـًٌٍَُِّْٱ]", "", text)
    text = re.sub(r"[^\wء-ي ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: str) -> list[str]:
    return [w for w in _normalise(value).split() if len(w) > 2 and w not in STOPWORDS]


def _domain_label(url: str) -> str:
    return AGGREGATOR_HOSTS.get(_host(url), _host(url))


def _social_channel(url: str) -> str:
    host = _host(url)
    if host == "facebook.com" or host.endswith("facebook.com") or host == "fb.watch":
        return "Facebook عام"
    if host == "x.com" or host.endswith("x.com") or host == "twitter.com" or host.endswith("twitter.com"):
        return "X / Twitter عام"
    return ""


def _rss_results(url: str, channel: str) -> list[SearchResult]:
    try:
        soup = BeautifulSoup(_request(url), "html.parser")
    except requests.RequestException:
        return []
    output: list[SearchResult] = []
    for item in soup.find_all("item"):
        def text(name: str) -> str:
            node = item.find(name)
            return node.get_text(" ", strip=True) if node else ""
        title = text("title")
        link = text("link")
        if not title or not link:
            continue
        description = text("description")
        source_node = item.find("source")
        source = source_node.get_text(" ", strip=True) if source_node else _domain_label(link)
        output.append(SearchResult(title=BeautifulSoup(title, "html.parser").get_text(" ", strip=True), url=link, source=source, published=text("pubDate"), snippet=BeautifulSoup(description, "html.parser").get_text(" ", strip=True)[:1200], channel=channel))
    return output


def _bing_results(query: str) -> list[SearchResult]:
    try:
        soup = BeautifulSoup(_request("https://www.bing.com/search?q=" + quote(query[:240])), "html.parser")
    except requests.RequestException:
        return []
    output: list[SearchResult] = []
    for card in soup.select("li.b_algo")[:12]:
        anchor = card.select_one("h2 a")
        if not anchor or not anchor.get("href"):
            continue
        snippet = card.select_one(".b_caption p")
        url = anchor["href"]
        output.append(SearchResult(title=anchor.get_text(" ", strip=True), url=url, source=_domain_label(url), published="", snippet=snippet.get_text(" ", strip=True) if snippet else "", channel="Bing Web", searched_query=query))
    return output


def _tavily_results(query: str) -> list[SearchResult]:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key or key.lower() in {"your tavily key", "مفتاح tavily الخاص بك", "ضع-المفتاح-هنا"}:
        return []
    try:
        response = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": query, "search_depth": "advanced", "max_results": 10, "include_answer": False, "include_raw_content": True}, timeout=30)
        response.raise_for_status()
        items = response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError):
        return []
    return [SearchResult(title=str(x.get("title", "")), url=str(x.get("url", "")), source=_domain_label(str(x.get("url", ""))), published=str(x.get("published_date", "")), snippet=str(x.get("raw_content") or x.get("content") or "")[:12000], channel="فهرس ويب موسع", searched_query=query) for x in items if x.get("url")]


def _ai_search_plan(claim: str) -> list[str]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return []
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    prompt = ("حلل الادعاء التالي كخبير بحث إخباري. أعد JSON بالمفتاح queries ويحتوي 8 استعلامات عربية مستقلة للعثور على نفس الحدث، لا أخبار عامة مشابهة. استخدم فقط الأسماء والأماكن والأفعال الموجودة في الادعاء، ولا تخترع معلومات. نوّع بين العبارة المميزة، الشخص/الجهة، المكان+الفعل، الفعل+الشخص، والنتيجة المذكورة. لا تستخدم site:sahaafa.net. الادعاء: " + claim)
    try:
        response = requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json={"model": os.getenv("NEWSGUARD_SEARCH_MODEL", "gpt-4o-mini"), "temperature": 0, "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]}, timeout=30)
        response.raise_for_status()
        data = json.loads(response.json()["choices"][0]["message"]["content"])
        return list(dict.fromkeys(str(q).strip() for q in data.get("queries", []) if str(q).strip()))[:8]
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return []


def _fallback_plan(claim: str) -> list[str]:
    words = _tokens(claim)
    values = [claim[:200]]
    if len(words) >= 4:
        values += [" ".join(words[:8]), " ".join(words[-8:]), " ".join(words[:5]), " ".join(words[-5:])]
    return list(dict.fromkeys(x for x in values if x.strip()))[:6]


def _extract_sahaafa_attribution(soup: BeautifulSoup, page_url: str) -> tuple[str, str]:
    """Return the publisher label and actual publisher href when explicitly present."""
    candidates: list[tuple[str, str]] = []
    marker_re = re.compile(r"المصدر\s*:?")
    for node in soup.find_all(string=marker_re):
        parent = getattr(node, "parent", None)
        if not parent:
            continue
        containers = [parent]
        if parent.parent:
            containers.append(parent.parent)
        for container in containers:
            for link in container.find_all("a", href=True):
                name = link.get_text(" ", strip=True)
                href = urljoin(page_url, str(link.get("href", "")))
                if name and "صحافة نت" not in name and _host(href) != "sahaafa.net":
                    candidates.append((name, href))
            if candidates:
                break
        if candidates:
            break
    if candidates:
        return candidates[0]
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
            if isinstance(publisher, dict) and publisher.get("name"):
                return str(publisher["name"]).strip()
            if isinstance(publisher, str) and publisher.strip():
                return publisher.strip()
    return ""


def _validate_attribution(result: SearchResult, soup: BeautifulSoup) -> SearchResult:
    host = _host(result.url)
    result.aggregator = AGGREGATOR_HOSTS.get(host, "")
    if host not in AGGREGATOR_HOSTS:
        publisher = _jsonld_publisher(soup)
        result.publisher_name = publisher or result.source or host
        result.publisher_url = _canonical_url(result.url)
        result.source = result.publisher_name
        result.attribution_status = "مباشر"
        return result

    name, href = _extract_sahaafa_attribution(soup, result.url)
    result.publisher_name = name
    result.publisher_url = _canonical_url(href) if href else ""
    result.source = name or "غير متحقق"
    result.attribution_status = "منسوب فقط" if name else "غير متحقق"

    if not result.publisher_url:
        return result
    if _host(result.publisher_url) == host:
        result.attribution_status = "غير متحقق"
        result.source = "غير متحقق"
        return result

    try:
        original_html = _request(result.publisher_url, timeout=10)
        original_soup = BeautifulSoup(original_html, "html.parser")
        og = original_soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            original_title = str(og.get("content"))
        else:
            h1 = original_soup.find("h1")
            original_title = h1.get_text(" ", strip=True) if h1 else (original_soup.title.get_text(" ", strip=True) if original_soup.title else "")
        title_score = SequenceMatcher(None, _normalise(result.title), _normalise(original_title)).ratio()
        original_text = original_soup.get_text(" ", strip=True)[:18000]
        title_tokens = set(_tokens(result.title))
        text_tokens = set(_tokens(original_text))
        overlap = len(title_tokens & text_tokens) / max(1, len(title_tokens))
        if title_score >= 0.72 or overlap >= 0.60:
            result.attribution_status = "متحقق"
        else:
            result.attribution_status = "غير متحقق"
            result.source = "غير متحقق"
    except requests.RequestException:
        result.attribution_status = "غير متحقق"
        result.source = "غير متحقق"
    return result


def _enrich_page(result: SearchResult) -> SearchResult:
    try:
        html = _request(result.url, timeout=12)
        soup = BeautifulSoup(html, "html.parser")
        if _host(result.url) in AGGREGATOR_HOSTS:
            result = _validate_attribution(result, soup)
        else:
            publisher = _jsonld_publisher(soup)
            if not result.source or result.source == _host(result.url):
                result.source = publisher or _domain_label(result.url)
            result.publisher_name = result.source
            result.publisher_url = _canonical_url(result.url)
            result.attribution_status = "مباشر"
        clean_soup = BeautifulSoup(html, "html.parser")
        for node in clean_soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        body = clean_soup.get_text(" ", strip=True)
        if body:
            result.snippet = (result.snippet + " " + body[:14000])[:16000]
        if not result.published:
            for attr in ["article:published_time", "datePublished", "pubdate"]:
                meta = clean_soup.find("meta", attrs={"property": attr}) or clean_soup.find("meta", attrs={"name": attr})
                if meta and meta.get("content"):
                    result.published = str(meta.get("content"))
                    break
    except requests.RequestException:
        pass
    return result


def _event_score(claim: str, result: SearchResult) -> float:
    claim_tokens = set(_tokens(claim))
    page_text = f"{result.title} {result.snippet}"
    page_tokens = set(_tokens(page_text))
    if not claim_tokens or not page_tokens:
        return 0.0
    overlap = len(claim_tokens & page_tokens) / len(claim_tokens)
    title_sim = SequenceMatcher(None, _normalise(claim), _normalise(result.title)).ratio()
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), max_features=5000).fit_transform([claim, page_text[:12000]])
        body_sim = float(cosine_similarity(matrix[0:1], matrix[1:2]).ravel()[0])
    except ValueError:
        body_sim = 0.0
    claim_locs = {x for x in claim_tokens if x in LOCATIONS}
    claim_actions = {x for x in claim_tokens if x in ACTIONS}
    location_overlap = len(claim_locs & page_tokens) / max(1, len(claim_locs)) if claim_locs else 0.0
    action_overlap = len(claim_actions & page_tokens) / max(1, len(claim_actions)) if claim_actions else 0.0
    score = overlap * 0.30 + title_sim * 0.25 + body_sim * 0.15 + location_overlap * 0.15 + action_overlap * 0.10
    normal_claim = _normalise(claim)
    normal_page = _normalise(page_text)
    query_words = _tokens(claim)
    for size in (8, 7, 6, 5):
        found = False
        for start in range(max(0, len(query_words) - size + 1)):
            phrase = " ".join(query_words[start:start + size])
            if phrase and phrase in normal_page:
                score += 0.20
                found = True
                break
        if found:
            break
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
        old = unique.get(key)
        if old is None or (result.attribution_status == "متحقق" and old.attribution_status != "متحقق") or len(result.snippet) > len(old.snippet):
            unique[key] = result
    return list(unique.values())


def _diversify(results: list[SearchResult], limit: int) -> list[SearchResult]:
    chosen: list[SearchResult] = []
    counts: dict[str, int] = {}
    regular_cap = max(3, min(5, limit // 5 if limit >= 15 else 3))
    for result in results:
        host = _host(result.url)
        cap = 3 if host in AGGREGATOR_HOSTS else regular_cap
        if counts.get(host, 0) >= cap:
            continue
        chosen.append(result)
        counts[host] = counts.get(host, 0) + 1
        if len(chosen) >= limit:
            break
    return chosen


def _user_provided_results(query: str) -> tuple[str, list[SearchResult]]:
    urls = re.findall(r"https?://[^\s<>]+", query or "")
    clean_query = re.sub(r"https?://[^\s<>]+", " ", query or "")
    results: list[SearchResult] = []
    for raw in dict.fromkeys(urls):
        url = raw.rstrip(".,،؛")
        result = SearchResult(title="رابط أدخله المستخدم", url=url, source=_host(url), published="", snippet="", channel=_social_channel(url) or "رابط مقدم من المستخدم", searched_query=clean_query.strip(), match_type="رابط مباشر مقدم من المستخدم")
        result = _enrich_page(result)
        if result.title == "رابط أدخله المستخدم":
            result.title = _host(url)
        results.append(result)
    return clean_query.strip(), results


def search_public_web(query: str, sources_path: Path, max_results: int = 25) -> list[dict]:
    query, direct = _user_provided_results(query)
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        return [asdict(x) for x in direct]

    variants = list(dict.fromkeys(_ai_search_plan(query) + _fallback_plan(query)))[:10]
    collected: list[SearchResult] = list(direct)
    for search_query in variants:
        collected.extend(_tavily_results(search_query))
        encoded = quote(search_query[:240])
        feeds = [
            (f"https://news.google.com/rss/search?q={encoded}&hl=ar&gl=YE&ceid=YE:ar", "Google News"),
            (f"https://www.bing.com/news/search?q={encoded}&format=rss", "Bing News"),
        ]
        for feed_url, channel in feeds:
            for item in _rss_results(feed_url, channel):
                item.searched_query = search_query
                collected.append(item)
        collected.extend(_bing_results(search_query))

    # Configured direct sources are supplementary. Aggregators are not crawled as a primary source list.
    if sources_path.exists():
        try:
            sources = pd.read_csv(sources_path).fillna("")
            for _, source in sources.iterrows():
                if str(source.get("active", "TRUE")).upper() != "TRUE":
                    continue
                base_url = str(source.get("url", "")).rstrip("/")
                if not base_url.startswith("http") or _host(base_url) in AGGREGATOR_HOSTS:
                    continue
                try:
                    soup = BeautifulSoup(_request(base_url, timeout=10), "html.parser")
                except requests.RequestException:
                    continue
                for link in soup.find_all("a", href=True)[:120]:
                    title = link.get_text(" ", strip=True)
                    if len(title) < 15 or len(title) > 300:
                        continue
                    collected.append(SearchResult(title=title, url=urljoin(base_url, link["href"]), source=str(source.get("name", _host(base_url))), published="", snippet="نتيجة من مصدر يمني مُهيأ", channel="مصدر يمني مباشر", searched_query=query))
        except (OSError, ValueError, TypeError):
            pass

    collected = _deduplicate(collected)
    # Enrich a bounded candidate pool to keep Streamlit responsive.
    collected.sort(key=lambda x: len(x.snippet), reverse=True)
    candidates = collected[:100]
    for index, item in enumerate(candidates):
        candidates[index] = _enrich_page(item)

    ranked: list[SearchResult] = []
    for item in candidates:
        item.match = _event_score(query, item)
        item.relation = _relation(item.match)
        item.evidence_score = item.match
        social = _social_channel(item.url)
        if social:
            item.channel = social
        if item.attribution_status == "متحقق":
            item.match_type = "مصدر أصلي متحقق — " + item.relation
        elif item.aggregator:
            item.match_type = "مجمع/فهرسة — مصدر منسوب" if item.match >= 0.55 else "مجمع/فهرسة — صلة ضعيفة"
        elif social:
            item.match_type = "منصة اجتماعية عامة — " + item.relation
        elif item.match >= 0.78:
            item.match_type = "تطابق قوي"
        elif item.match >= 0.55:
            item.match_type = "سياق مرتبط"
        else:
            item.match_type = "غير ذي صلة"
        if item.match >= 0.50 or (social and item.match >= 0.35):
            ranked.append(item)

    ranked.sort(key=lambda x: (x.match, x.attribution_status == "متحقق", bool(x.publisher_url)), reverse=True)
    return [asdict(x) for x in _diversify(ranked, max_results)]
