"""Public web/news discovery for claim comparison.

This searches indexed public web/news feeds and configured public source pages.
It does not bypass logins, private posts, paywalls, or platform API restrictions.
"""
from __future__ import annotations

import re
import warnings
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit
from difflib import SequenceMatcher

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; YemenNewsGuard/1.1; +https://github.com/Muoz22/yemen-newsguard-ai)"}
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


def _request(url: str, timeout: int = 15) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _rss_results(url: str, channel: str) -> list[SearchResult]:
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
    """Parse public Bing result cards, including site-restricted searches."""
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
    """Search configured source homepages and paginated public indexes.

    The previous implementation inspected only the current homepage. That is
    insufficient for aggregators such as Sahaafa: a story can be published,
    indexed, and still be absent from today's first page. We therefore inspect
    a bounded set of public pagination pages and rank their visible headlines.
    """
    if not sources_path.exists():
        return []
    sources = pd.read_csv(sources_path).fillna("")
    terms = [term for term in re.findall(r"[\wء-ي]{3,}", query.lower()) if term not in {"من", "في", "على", "هذا", "هذه", "التي", "الذي", "عن", "إلى", "وقد", "كما"}]
    results: list[SearchResult] = []

    for _, source in sources.iterrows():
        if str(source.get("active", "TRUE")).upper() != "TRUE":
            continue
        base_url = str(source.get("url", "")).rstrip("/")
        if not base_url.startswith("http"):
            continue

        pages = [base_url]
        host = urlsplit(base_url).netloc.lower()
        # Sahaafa exposes a simple page1.html, page2.html ... public index.
        if "sahaafa.net" in host:
            pages.extend([f"{base_url}/page{i}.html" for i in range(1, 11)])
            pages.extend([f"{base_url}/topic{i}.html" for i in range(1, 8)])

        seen_pages = set()
        for page_url in pages:
            if page_url in seen_pages:
                continue
            seen_pages.add(page_url)
            try:
                soup = BeautifulSoup(_request(page_url, timeout=12), "html.parser")
            except requests.RequestException:
                continue

            for link in soup.find_all("a", href=True):
                title = link.get_text(" ", strip=True)
                if len(title) < 12 or len(title) > 300:
                    continue
                haystack = title.lower()
                overlap = sum(term in haystack for term in terms)
                # Keep only meaningful candidates. For short queries, one exact
                # keyword can be enough; for long claims require broader overlap.
                threshold = 1 if len(terms) <= 3 else max(2, int(len(terms) * 0.25))
                if overlap < threshold:
                    continue
                article_url = urljoin(page_url, link["href"])
                results.append(SearchResult(
                    title=title,
                    url=article_url,
                    source=str(source.get("name", base_url)),
                    published="",
                    snippet="نتيجة من الفهرس العام للمصدر",
                    match=min(0.95, overlap / max(1, len(terms))),
                    channel="مصدر يمني مباشر",
                ))
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

    for size in (8, 7, 6, 5):
        for start in range(max(0, len(query_words) - size + 1)):
            phrase = " ".join(query_words[start:start + size])
            if phrase in right:
                return 0.98

    overlap = len(set(query_words) & page_words) / len(set(query_words))
    title = normalize(result.title)
    title_similarity = SequenceMatcher(None, left, title).ratio()
    score = overlap * 0.72 + title_similarity * 0.28
    return round(score, 2) if overlap >= 0.65 else round(score * 0.45, 2)


def _enrich_page(result: SearchResult) -> SearchResult:
    """Read a public result page so matching uses article text, not only a headline."""
    try:
        html = _request(result.url, timeout=10)
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
    if len(words) >= 4:
        variants.append(" ".join(words[-6:]))
    ai_variants = _ai_search_queries(query)
    variants = ai_variants + variants
    variants = list(dict.fromkeys(v for v in variants if v.strip()))[:5]
    collected: list[SearchResult] = list(direct_results)

    # Tavily gets both broad and exact/platform-scoped queries when configured.
    tavily_queries = [
        query,
        *ai_variants[:3],
        f'"{query}"',
        f'site:sahaafa.net "{query}"',
        f'site:facebook.com "{query}"',
        f'site:x.com "{query}"',
        f'site:twitter.com "{query}"',
    ]
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
        collected.extend(_bing_html_results(f'site:sahaafa.net "{search_query[:180]}"', "صحافة نت عبر Bing Web"))

        found = _source_page_results(sources_path, search_query)
        for result in found:
            result.searched_query = search_query
        collected.extend(found)

    unique: dict[str, SearchResult] = {}
    for result in collected:
        if result.url and result.url not in unique:
            unique[result.url] = result

    # Enrich more than the old fixed first-20 slice. Source-specific hits must
    # be read before final scoring, otherwise an older Sahaafa page can be
    # discarded merely because it was discovered later in the run.
    candidates = list(unique.values())[:80]
    for index, item in enumerate(candidates):
        candidates[index] = _enrich_page(item)
    unique.update({item.url: item for item in candidates})

    ranked = _similarity(query, list(unique.values()))
    for item in ranked:
        item.match = _exactness(query, item)
        social = _social_channel(item.url)
        direct = item.match_type == "رابط مباشر مقدم من المستخدم"
        if direct:
            item.match_type = "رابط مباشر مقدم من المستخدم — تعذر/تمكن قراءة المحتوى"
        elif social:
            item.channel = social
            item.match_type = "مطابق/مشابه من منصة اجتماعية"
        elif item.match >= 0.75:
            item.match_type = "تطابق قوي"
        else:
            item.match_type = "تطابق جزئي"

    ranked.sort(key=lambda item: item.match, reverse=True)
    ranked = [item for item in ranked if item.match >= 0.60 or (_social_channel(item.url) and item.match >= 0.35) or item.match_type.startswith("رابط مباشر مقدم من المستخدم")]
    return [asdict(item) for item in ranked[:max_results]]
