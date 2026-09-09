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
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
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
    """Extract links from a webpage."""
    html = fetch_page(url)

    soup = BeautifulSoup(html, "html.parser")

    links = []

    for link in soup.find_all("a", href=True):

        href = urljoin(url, link["href"])

        title = link.get_text(" ", strip=True)

        if title and href.startswith("http"):
            links.append({
                "title": title,
                "url": href
            })

    return links


def main():

    sources = load_sources()

    print("Configured sources:")
    print(sources[[
        "source_id",
        "name",
        "url",
        "source_type",
        "source_role"
    ]].to_string(index=False))

    print("\nCollector initialized successfully.")


if __name__ == "__main__":
    main()
