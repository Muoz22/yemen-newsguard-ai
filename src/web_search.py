"""Public web/news discovery for claim comparison.

This searches indexed public web/news feeds and configured public source pages.
It does not bypass logins, private posts, paywalls, or platform API restrictions.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urljoin

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


def _request(url: str, timeout: int = 15) -> str:
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


def search_public_web(query: str, sources_path: Path, max_results: int = 25) -> list[dict]:
    """Search public indexed news and configured public pages, deduplicated by URL."""
    query = re.sub(r"\s+", " ", (query or "")).strip()
    if not query:
        return []
    words = [w for w in re.findall(r"[\wء-ي]{3,}", query.lower()) if w not in {"من", "في", "على", "هذا", "هذه", "التي", "الذي", "عن", "إلى", "وقد", "كما"}]
    variants = [query[:180]]
    if len(words) >= 4:
        variants.append(" ".join(words[:8]))
    # Search the event, place, and actors separately so a copied social post
    # does not need to match a publisher's headline word for word.
    if len(words) >= 4:
        variants.append(" ".join(words[-6:]))
    variants = list(dict.fromkeys(v for v in variants if v.strip()))
    collected: list[SearchResult] = []
    for search_query in variants:
        encoded = quote(search_query[:240])
        feeds = [
            (f"https://news.google.com/rss/search?q={encoded}&hl=ar&gl=YE&ceid=YE:ar", "Google News"),
            (f"https://www.bing.com/news/search?q={encoded}&format=rss", "Bing News"),
        ]
        for url, channel in feeds:
            try:
                found = _rss_results(url, channel)
                for result in found:
                    result.searched_query = search_query
                collected.extend(found)
            except requests.RequestException:
                continue
        found = _source_page_results(sources_path, search_query)
        for result in found:
            result.searched_query = search_query
        collected.extend(found)
    unique: dict[str, SearchResult] = {}
    for result in collected:
        if result.url and result.url not in unique:
            unique[result.url] = result
    ranked = _similarity(query, list(unique.values()))
    # Keep weakly related headlines out of the evidence table. A low score is
    # not evidence that the claim was published; it is only a search lead.
    ranked = [item for item in ranked if item.match >= 0.20]
    return [asdict(item) for item in ranked[:max_results]]
