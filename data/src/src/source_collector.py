import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_FILE = BASE_DIR / "data" / "sources.csv"


def load_sources():
    """Load configured news sources."""
    return pd.read_csv(SOURCES_FILE)


def fetch_page(url):
    """Download a webpage."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()
    return response.text


def extract_links(url):
    """Extract article-like links from a webpage."""

    html = fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")

    links = []
    seen = set()

    for link in soup.find_all("a", href=True):

        href = urljoin(url, link["href"])
        title = link.get_text(" ", strip=True)

        if not title:
            continue

        if not href.startswith("http"):
            continue

        if href in seen:
            continue

        seen.add(href)

        links.append({
            "title": title,
            "url": href
        })

    return links


def test_source(source_id, name, url):

    print("\n" + "=" * 60)
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print("=" * 60)

    try:

        links = extract_links(url)

        print(f"Successfully extracted {len(links)} links.")

        for item in links[:10]:
            print(f"- {item['title']}")
            print(f"  {item['url']}")

        return True

    except Exception as e:

        print(f"ERROR: {e}")
        return False


def main():

    sources = load_sources()

    print("\nYemen NewsGuard AI")
    print("Source Collector Test")
    print("=" * 60)

    success = 0
    failed = 0

    for _, source in sources.iterrows():

        result = test_source(
            source["source_id"],
            source["name"],
            source["url"]
        )

        if result:
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print(f"Successful sources: {success}")
    print(f"Failed sources: {failed}")
    print(f"Total sources: {len(sources)}")


if __name__ == "__main__":
    main()
