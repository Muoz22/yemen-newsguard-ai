"""Public web/news discovery for claim comparison.

This searches indexed public web/news feeds and configured public source pages.
It does not bypass logins, private posts, paywalls, or platform API restrictions.
"""
from __future__ import annotations

import re
import warnings
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit
from difflib import SequenceMatcher

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; YemenNewsGuard/1.0; +https://github.com/Muoz22/yemen-newsguard-ai)"}
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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
    relation: str = ""
    attributed_source: str = ""
    publisher_type: str = ""


def _request(url: str, timeout: int = 8) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _rss_results(url: str, channel: str) -> list[SearchResult]:
    # html.parser ships with Python and also handles the simple RSS tags we need.
    # Using the optional XML parser caused FeatureNotFound on Streamlit Cloud.
    soup = BeautifulSoup(_request(url), "html.parser")
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
        source = source_node.get_text(" ", strip=True) if source_node else channel
        if title and link:
            results.append(SearchResult(title=BeautifulSoup(title, "html.parser").get_text(" ", strip=True), url=link,
                                        source=source, published=date or "", snippet=BeautifulSoup(description or "", "html.parser").get_text(" ", strip=True)[:300], channel=channel))
    return results


def _bing_html_results(query: str, channel: str = "Bing Web") -> list[SearchResult]:
    """Parse public Bing result cards when RSS omits a short headline."""
    try:
        soup = BeautifulSoup(_request("https://www.bing.com/search?q=" + quote(query[:220])), "html.parser")
        results: list[SearchResult] = []
        for card in soup.select("li.b_algo")[:10]:
            anchor = card.select_one("h2 a")
            if not anchor or not anchor.get("href"):
                continue
            snippet_node = card.select_one(".b_caption p")
            results.append(SearchResult(title=anchor.get_text(" ", strip=True), url=anchor["href"], source=urlsplit(anchor["href"]).netloc,
                                        published="", snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "", channel=channel, searched_query=query))
        return results
    except requests.RequestException:
        return []


def _source_page_results(sources_path: Path, query: str) -> list[SearchResult]:
    if not sources_path.exists():
        return []
    sources = pd.read_csv(sources_path).fillna("")
    terms = [term for term in re.findall(r"[\wء-ي]{3,}", query.lower()) if term not in {"من", "في", "على", "هذا", "التي", "الذي"}]
    results: list[SearchResult] = []
    for _, source in sources.iterrows():
        if str(source.get("active", "TRUE")).upper() != "TRUE":
            continue
        url = str(source.get("url", ""))
        if not url.startswith("http"):
            continue
        try:
            soup = BeautifulSoup(_request(url), "html.parser")
            for link in soup.find_all("a", href=True):
                title = link.get_text(" ", strip=True)
                if len(title) < 18 or len(title) > 250:
                    continue
                haystack = title.lower()
                overlap = sum(term in haystack for term in terms)
                if overlap or not terms:
                    results.append(SearchResult(title=title, url=urljoin(url, link["href"]), source=str(source.get("name", url)),
                                                published="", snippet="نتيجة من صفحة المصدر العامة", match=min(0.99, overlap / max(1, len(terms))), channel="مصدر يمني مباشر"))
        except requests.RequestException:
            continue
    return results


def _similarity(query: str, results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return []
    texts = [query] + [f"{r.title} {r.snippet}" for r in results]
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), max_features=5000).fit_transform(texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    except ValueError:
        scores = [0.0] * len(results)
    for result, score in zip(results, scores):
        result.match = round(max(result.match, float(score)), 2)
    return sorted(results, key=lambda r: r.match, reverse=True)


def _tavily_results(query: str) -> list[SearchResult]:
    """Use a real web index when TAVILY_API_KEY is configured."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.strip().lower() in {"مفتاح tavily الخاص بك", "your tavily key", "ضع-المفتاح-هنا"}:
        return []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "search_depth": "advanced", "max_results": 10,
                  "include_answer": False, "include_raw_content": True},
            timeout=30,
        )
        response.raise_for_status()
        items = response.json().get("results", [])
        return [SearchResult(title=str(item.get("title", "")), url=str(item.get("url", "")),
                             source=str(item.get("url", "")).split("/")[2] if item.get("url") else "ويب عام",
                             published=str(item.get("published_date", "")), snippet=str(item.get("raw_content") or item.get("content", ""))[:10000],
                             channel="فهرس ويب موسع", searched_query=query) for item in items if item.get("url")]
    except (requests.RequestException, ValueError, TypeError):
        return []


def _ai_search_queries(claim: str) -> list[str]:
    """Turn a long social post into precise searchable claim variants."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    prompt = "استخرج 5 عبارات بحث عربية دقيقة للعثور على نفس الخبر المنشور، لا أخباراً عن موضوع مشابه. أعد JSON فقط بالمفتاح queries، واجعل إحدى العبارات جملة مميزة من 8 إلى 14 كلمة. لا تضف معلومات غير موجودة. الادعاء: " + claim
    try:
        response = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": os.getenv("NEWSGUARD_SEARCH_MODEL", "gpt-4o-mini"), "temperature": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        response.raise_for_status()
        data = __import__("json").loads(response.json()["choices"][0]["message"]["content"])
        return [str(q).strip() for q in data.get("queries", []) if str(q).strip()][:5]
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return []


def _exactness(query: str, result: SearchResult) -> float:
    """Score whether a page contains the same claim, not merely the same topic."""
    stopwords = {"من", "في", "على", "هذا", "هذه", "التي", "الذي", "عن", "إلى", "وقد", "كما", "أن", "إن", "تم", "مع", "بعد", "قبل"}

    def normalize(value: str) -> str:
        value = str(value or "").lower().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
        value = re.sub(r"[^\wء-ي ]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    left = normalize(query)
    right = normalize(f"{result.title} {result.snippet}")
    query_words = [word for word in left.split() if len(word) > 2 and word not in stopwords]
    page_words = set(right.split())
    if not query_words or not page_words:
        return 0.0

    # An uninterrupted six-word phrase is strong evidence of a copied post.
    for size in (8, 7, 6, 5):
        for start in range(max(0, len(query_words) - size + 1)):
            phrase = " ".join(query_words[start:start + size])
            if phrase in right:
                return 0.98

    overlap = len(set(query_words) & page_words) / len(set(query_words))
    title = normalize(result.title)
    title_similarity = SequenceMatcher(None, left, title).ratio()
    # Related stories normally share only actors or a location; require broad
    # claim coverage before calling a result a match.
    score = overlap * 0.72 + title_similarity * 0.28
    return round(score, 2) if overlap >= 0.65 else round(score * 0.45, 2)


def _enrich_page(result: SearchResult) -> SearchResult:
    """Read a public result page so matching uses article text, not only a headline."""
    try:
        html = _request(result.url, timeout=4)
        soup = BeautifulSoup(html, "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        body = soup.get_text(" ", strip=True)
        if body:
            result.snippet = f"{result.snippet} {body[:10000]}"[:11000]
    except requests.RequestException:
        pass
    return result


def _social_channel(url: str) -> str:
    host = url.lower()
    if "facebook.com" in host or "fb.watch" in host:
        return "Facebook عام"
    if "x.com" in host or "twitter.com" in host:
        return "X / Twitter عام"
    return ""


def _publisher_type(result: SearchResult) -> tuple[str, str]:
    text = f"{result.title} {result.snippet}"
    attributed = ""
    found = re.search(r"(?:نقلاً عن|نقلا عن|بحسب|المصدر|عن مصدر|نقلًا عن)\s*[:：]?\s*([\wء-ي .-]{2,60})", text, re.I)
    if found:
        attributed = found.group(1).strip(" .،؛:-")
    domain = urlsplit(result.url).netloc.lower()
    if any(x in domain for x in ("sahaafa.net", "klyoum", "pressbee", "msader", "arabic.pressbee")):
        kind = "ناقل/مجمع محتمل"
    elif _social_channel(result.url):
        kind = "منصة اجتماعية"
    else:
        kind = "ناشر ويب"
    return kind, attributed


def _user_provided_results(query: str) -> tuple[str, list[SearchResult]]:
    """Inspect URLs pasted with the claim and keep them visible even if blocked."""
    urls = re.findall(r"https?://[^\s<>]+", query or "")
    clean_query = re.sub(r"https?://[^\s<>]+", " ", query or "")
    results: list[SearchResult] = []
    for raw_url in dict.fromkeys(urls):
        url = raw_url.rstrip(".,،؛")
        host = urlsplit(url).netloc.lower()
        channel = _social_channel(url) or "رابط مقدم من المستخدم"
        result = SearchResult(title="رابط أدخله المستخدم", url=url, source=host, published="", snippet="",
                              match=0.0, channel=channel, searched_query=clean_query.strip(), match_type="رابط مباشر مقدم من المستخدم")
        try:
            result = _enrich_page(result)
            if result.title == "رابط أدخله المستخدم":
                result.title = host
        except Exception:
            pass
        results.append(result)
    return clean_query.strip(), results


def search_public_web(query: str, sources_path: Path, max_results: int = 25) -> list[dict]:
    """Search public indexed news and configured public pages, deduplicated by URL."""
    query, direct_results = _user_provided_results(query)
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        return [asdict(item) for item in direct_results]
    words = [w for w in re.findall(r"[\wء-ي]{3,}", query.lower()) if w not in {"من", "في", "على", "هذا", "هذه", "التي", "الذي", "عن", "إلى", "وقد", "كما"}]
    variants = [query[:180]]
    if len(words) >= 4:
        variants.append(" ".join(words[:8]))
    # Search the event, place, and actors separately so a copied social post
    # does not need to match a publisher's headline word for word.
    if len(words) >= 4:
        variants.append(" ".join(words[-6:]))
    ai_variants = _ai_search_queries(query)
    variants = ai_variants + variants
    variants = list(dict.fromkeys(v for v in variants if v.strip()))[:5]
    collected: list[SearchResult] = list(direct_results)
    # Exact and platform-scoped searches reduce unrelated topic matches.
    tavily_queries = [
        query,
        *ai_variants[:3],
        f'"{query}"',
        f'site:sahaafa.net "{query}"',
        f'site:facebook.com "{query}"',
        f'site:x.com "{query}"',
        f'site:twitter.com "{query}"',
    ]
    if sources_path.exists():
        try:
            domains = []
            for raw in pd.read_csv(sources_path).fillna("").get("url", []):
                domain = urlsplit(str(raw)).netloc.replace("www.", "")
                if domain and domain not in domains:
                    domains.append(domain)
            for domain in domains[:20]:
                tavily_queries.append(f'site:{domain} "{query}"')
        except (OSError, ValueError):
            pass
    for tavily_query in tavily_queries:
        collected.extend(_tavily_results(tavily_query))
    for search_query in variants:
        encoded = quote(search_query[:240])
        feeds = [
            (f"https://news.google.com/rss/search?q={encoded}&hl=ar&gl=YE&ceid=YE:ar", "Google News"),
            (f"https://www.bing.com/news/search?q={encoded}&format=rss", "Bing News"),
            (f"https://news.google.com/rss/search?q={quote('site:sahaafa.net ' + search_query[:180])}&hl=ar&gl=YE&ceid=YE:ar", "صحافة نت عبر فهرس الأخبار"),
            (f"https://www.bing.com/news/search?q={quote('site:sahaafa.net ' + search_query[:180])}&format=rss", "صحافة نت عبر Bing"),
            (f"https://news.google.com/rss/search?q={quote('site:facebook.com ' + search_query[:180])}&hl=ar&gl=YE&ceid=YE:ar", "Facebook عام عبر فهرس الأخبار"),
            (f"https://news.google.com/rss/search?q={quote('site:x.com ' + search_query[:180])}&hl=ar&gl=YE&ceid=YE:ar", "X عام عبر فهرس الأخبار"),
        ]
        for url, channel in feeds:
            try:
                found = _rss_results(url, channel)
                for result in found:
                    result.searched_query = search_query
                collected.extend(found)
            except requests.RequestException:
                continue
        collected.extend(_bing_html_results(search_query))
        if sources_path.exists() and search_query == variants[0]:
            try:
                for raw in pd.read_csv(sources_path).fillna("").get("url", []):
                    domain = urlsplit(str(raw)).netloc.replace("www.", "")
                    if domain:
                        collected.extend(_bing_html_results(f"site:{domain} {search_query}"))
            except (OSError, ValueError):
                pass
        found = _source_page_results(sources_path, search_query)
        for result in found:
            result.searched_query = search_query
        collected.extend(found)
    unique: dict[str, SearchResult] = {}
    for result in collected:
        if result.url and result.url not in unique:
            unique[result.url] = result
    # Search providers often put generic stories first. Keep a broad candidate
    # pool so a relevant domain is not discarded before page verification.
    candidates = list(unique.values())[:50]
    for index, item in enumerate(candidates):
        candidates[index] = _enrich_page(item)
    unique.update({item.url: item for item in candidates})
    ranked = _similarity(query, list(unique.values()))
    for item in ranked:
        item.match = _exactness(query, item)
        social = _social_channel(item.url)
        item.publisher_type, item.attributed_source = _publisher_type(item)
        direct = item.match_type == "رابط مباشر مقدم من المستخدم"
        if direct:
            item.match_type = "رابط مباشر مقدم من المستخدم — تعذر/تمكن قراءة المحتوى"
        elif social:
            item.channel = social
            item.match_type = "تطابق/سياق مرتبط من منصة اجتماعية"
        elif item.match >= 0.75:
            item.match_type = "تطابق قوي"
        else:
            item.match_type = "تطابق جزئي"
        item.relation = "نفس الخبر/إعادة نشر محتملة" if item.match >= 0.75 else ("نفس الحدث أو سياق مرتبط" if item.match >= 0.35 else "موضوع قريب")
    ranked.sort(key=lambda item: item.match, reverse=True)
    # Keep weakly related headlines out of the evidence table. A low score is
    # not evidence that the claim was published; it is only a search lead.
    ranked = [item for item in ranked if item.match >= 0.60 or item.match_type.startswith("رابط مباشر مقدم من المستخدم") or (item.relation == "نفس الحدث أو سياق مرتبط" and item.publisher_type != "ناشر ويب")]
    return [asdict(item) for item in ranked[:max_results]]
