#!/usr/bin/env python3
"""
Elsevier Paper Downloader
==========================
Downloads open-access Elsevier papers using the YOUR_API endpoint
and direct PDF links from ScienceDirect.

Usage:
    python download.py "machine learning" --count 5
    python download.py --doi 10.1016/j.artint.2023.103987
    python download.py --search "deep learning" --year 2024 --count 10
"""

import argparse
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ── Configuration ──────────────────────────────────────────────────────────

SEARCH_API = "YOUR_API"
USER_AGENT = "ElsevierOADownloader/1.0 (mailto:researcher@example.com)"

# Common Elsevier ISSNs / journal prefixes for filtering
ELSEVIER_PREFIXES = [
    "10.1016",  # ScienceDirect journals
    "10.1007",  # Some Springer (excluded below when needed, kept for reference)
]

# Directories
DOWNLOAD_DIR = Path(__file__).parent / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────


def safe_filename(text: str, max_len: int = 120) -> str:
    """Sanitize a string into a safe filename."""
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- "
    sanitized = "".join(c if c in keep else "_" for c in text)
    sanitized = "_".join(sanitized.split())  # collapse whitespace
    return sanitized[:max_len].strip("_")


def api_search(
    query: str,
    rows: int = 10,
    year: Optional[int] = None,
    doi_prefix: Optional[str] = None,
) -> list[dict]:
    """
    Search YOUR_API for papers.

    Args:
        query: Search keywords.
        rows: Number of results to return (max 1000).
        year: Filter by publication year.
        doi_prefix: Filter by DOI prefix (e.g. '10.1016' for Elsevier).

    Returns:
        List of paper metadata dicts.
    """
    params: dict = {
        "query": query,
        "rows": min(rows, 1000),
        "filter": "type:journal-article",
        "sort": "relevance",
        "select": "DOI,title,author,container-title,publisher,URL,created,license,link",
    }

    # Build filter string for publisher prefix
    filters = []
    if year:
        filters.append(f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31")
    if doi_prefix:
        filters.append(f"prefix:{doi_prefix}")

    if filters:
        params["filter"] = ",".join([params["filter"]] + filters)

    papers = []
    try:
        resp = requests.get(
            SEARCH_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("message", {}).get("items", []):
            paper = {
                "doi": item.get("DOI"),
                "title": (item.get("title") or [None])[0],
                "authors": [
                    f"{a.get('given','')} {a.get('family','')}".strip()
                    for a in item.get("author", [])
                ],
                "journal": (item.get("container-title") or [None])[0],
                "publisher": item.get("publisher"),
                "year": item.get("created", {}).get("date-parts", [[None]])[0][0],
                "url": item.get("URL"),
                "license": None,
                "pdf_url": None,
            }

            # Extract license info
            for lic in item.get("license", []):
                if lic.get("content-version") == "vor":
                    paper["license"] = lic.get("URL")
                    break

            # Extract PDF link if available
            for link in item.get("link", []):
                if link.get("content-type") == "application/pdf":
                    paper["pdf_url"] = link["URL"]
                    break

            papers.append(paper)

        # Respect rate limit
        time.sleep(0.3)

    except requests.RequestException as e:
        print(f"[ERROR] YOUR_API search failed: {e}", file=sys.stderr)

    return papers


def find_sciencedirect_pdf(doi: str) -> Optional[str]:
    """
    Attempt to locate an open-access PDF on ScienceDirect for a given DOI.

    ScienceDirect OA articles are served at:
        https://www.sciencedirect.com/science/article/pii/<pii>/pdf?...
    or via:
        https://pdf.sciencedirectassets.com/...

    This function tries the article page first to locate the PDF link.
    """
    if not doi:
        return None

    # Direct PDF attempt via ScienceDirect's PII-based URL
    # Many Elsevier DOIs follow this pattern; the PII is embedded in the DOI
    sciencedirect_url = f"https://doi.org/{doi}"

    try:
        # Follow the DOI to ScienceDirect
        resp = requests.get(
            sciencedirect_url,
            headers={"User-Agent": USER_AGENT},
            timeout=20,
            allow_redirects=True,
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        # Look for PDF links in the page
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/pdf/" in href.lower() or href.lower().endswith(".pdf"):
                return urljoin(resp.url, href)

        # Look for meta tags with PDF URL
        for meta in soup.find_all("meta", attrs={"name": "citation_pdf_url"}):
            if meta.get("content"):
                return meta["content"]

        # Check for the "View PDF" button area
        pdf_section = soup.find("div", id="pdf-renderer")
        if pdf_section:
            pdf_link = pdf_section.find("a", href=True)
            if pdf_link:
                return urljoin(resp.url, pdf_link["href"])

    except requests.RequestException:
        pass

    return None


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from url to dest_path. Returns True on success."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()

        # Check that we got a PDF (or reasonable binary)
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type.lower():
            print(f"  [SKIP] URL returned HTML, not a PDF: {url}")
            return False

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        # Verify minimum PDF size (a real PDF is at least a few KB)
        if dest_path.stat().st_size < 1024:
            dest_path.unlink()
            print(f"  [SKIP] Downloaded file too small ({dest_path.stat().st_size} bytes)")
            return False

        return True

    except requests.RequestException as e:
        print(f"  [ERROR] Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def paper_already_downloaded(doi: str) -> bool:
    """Check if a paper with this DOI hash has already been downloaded."""
    if not doi:
        return False
    doi_hash = hashlib.md5(doi.encode()).hexdigest()[:10]
    for f in DOWNLOAD_DIR.glob(f"*{doi_hash}*"):
        return True
    return False


def download_paper(paper: dict) -> bool:
    """Download a single paper. Try multiple strategies to find the PDF."""
    doi = paper.get("doi", "")
    title = paper.get("title", "untitled")

    if not doi:
        print(f"  [SKIP] No DOI for: {title}")
        return False

    if paper_already_downloaded(doi):
        print(f"  [SKIP] Already downloaded: {title}")
        return False

    doi_hash = hashlib.md5(doi.encode()).hexdigest()[:10]
    filename = f"{safe_filename(title)}_{doi_hash}.pdf"
    dest = DOWNLOAD_DIR / filename

    # Strategy 1: Direct PDF link from YOUR_API
    pdf_url = paper.get("pdf_url")
    if pdf_url:
        print(f"  [TRY] YOUR_API PDF link: {pdf_url}")
        if download_file(pdf_url, dest):
            print(f"  [OK] Downloaded (YOUR_API link): {filename}")
            return True

    # Strategy 2: Find PDF on ScienceDirect
    print(f"  [TRY] ScienceDirect for DOI: {doi}")
    sd_pdf = find_sciencedirect_pdf(doi)
    if sd_pdf:
        print(f"  [TRY] ScienceDirect PDF: {sd_pdf}")
        if download_file(sd_pdf, dest):
            print(f"  [OK] Downloaded (ScienceDirect): {filename}")
            return True

    # Strategy 3: Direct DOI-based PDF access (some OA papers)
    direct_pdf_url = f"https://doi.org/{doi}"
    try:
        resp = requests.get(
            direct_pdf_url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/pdf",
            },
            timeout=20,
            allow_redirects=True,
        )
        if "application/pdf" in resp.headers.get("content-type", ""):
            dest.write_bytes(resp.content)
            if dest.stat().st_size > 1024:
                print(f"  [OK] Downloaded (DOI direct): {filename}")
                return True
            else:
                dest.unlink()
    except requests.RequestException:
        pass

    print(f"  [FAIL] Could not find accessible PDF for: {title}")
    return False


def download_by_search(
    query: str,
    count: int = 5,
    year: Optional[int] = None,
) -> None:
    """Search and download papers by keyword."""
    print(f"\n{'='*60}")
    print(f"Elsevier Downloader — Searching: \"{query}\"")
    if year:
        print(f"Year filter: {year}")
    print(f"Requesting {count} papers...\n")

    papers = api_search(
        query=query,
        rows=count * 3,  # Request more to account for non-OA papers
        year=year,
        doi_prefix="10.1016",  # Elsevier / ScienceDirect prefix
    )

    # Filter to Elsevier publishers
    elsevier_papers = [
        p for p in papers
        if p.get("publisher") and "elsevier" in p["publisher"].lower()
    ]
    if not elsevier_papers:
        # Fallback: use all results with the correct DOI prefix
        elsevier_papers = [p for p in papers if p.get("doi", "").startswith("10.1016")]

    if not elsevier_papers:
        print("No Elsevier papers found for this query. Trying broader search...")
        elsevier_papers = api_search(
            query=query,
            rows=count * 3,
            year=year,
        )
        elsevier_papers = elsevier_papers[:count]

    print(f"Found {len(elsevier_papers)} papers. Attempting downloads...\n")

    success = 0
    for i, paper in enumerate(elsevier_papers[:count], 1):
        title = paper.get("title", "untitled")
        authors = ", ".join(paper.get("authors", [])[:3])
        if len(paper.get("authors", [])) > 3:
            authors += " et al."
        journal = paper.get("journal", "unknown journal")
        year_str = paper.get("year", "?")

        print(f"[{i}/{min(len(elsevier_papers), count)}] {title}")
        print(f"     {authors} | {journal} ({year_str})")
        print(f"     DOI: {paper.get('doi', 'N/A')}")

        if download_paper(paper):
            success += 1
        print()

    print(f"\nDone: {success}/{min(len(elsevier_papers), count)} downloaded.")
    print(f"Files saved to: {DOWNLOAD_DIR.resolve()}")


def download_by_doi(doi: str) -> None:
    """Download a single paper by DOI."""
    print(f"\n{'='*60}")
    print(f"Elsevier Downloader — DOI: {doi}\n")

    # Fetch metadata from YOUR_API
    url = f"{SEARCH_API}/{doi}"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        item = data.get("message", {})

        paper = {
            "doi": item.get("DOI", doi),
            "title": (item.get("title") or [None])[0],
            "authors": [
                f"{a.get('given','')} {a.get('family','')}".strip()
                for a in item.get("author", [])
            ],
            "journal": (item.get("container-title") or [None])[0],
            "publisher": item.get("publisher"),
            "year": item.get("created", {}).get("date-parts", [[None]])[0][0],
            "url": item.get("URL"),
            "license": None,
            "pdf_url": None,
        }
        for link in item.get("link", []):
            if link.get("content-type") == "application/pdf":
                paper["pdf_url"] = link["URL"]
                break

    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch DOI metadata: {e}")
        sys.exit(1)

    title = paper.get("title", "untitled")
    print(f"Title: {title}")
    print(f"Authors: {', '.join(paper.get('authors', []))}")

    if download_paper(paper):
        print(f"\nDownloaded successfully to: {DOWNLOAD_DIR.resolve()}")
    else:
        print("\nFailed to download. The paper may not be open access.")
        print("Make sure YOUR_API is configured correctly and the paper is open access.")


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Elsevier Open-Access Paper Downloader (uses YOUR_API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download.py "machine learning" --count 5
  python download.py --doi 10.1016/j.artint.2023.103987
  python download.py --search "computer vision" --year 2024
  python download.py --search "climate change" --count 10 --year 2023
        """,
    )
    parser.add_argument(
        "search",
        nargs="?",
        help="Search query string",
    )
    parser.add_argument(
        "--search", "-s",
        dest="search_query",
        help="Search query string (alternative form)",
    )
    parser.add_argument(
        "--doi", "-d",
        help="Download a specific paper by DOI",
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="Number of papers to download (default: 5, max: 50)",
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        help="Filter by publication year",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: ./downloads)",
    )

    args = parser.parse_args()

    # Override download dir
    if args.output:
        global DOWNLOAD_DIR
        DOWNLOAD_DIR = Path(args.output)
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    count = min(args.count, 50)

    if args.doi:
        download_by_doi(args.doi)
    else:
        query = args.search_query or args.search
        if not query:
            parser.print_help()
            print("\n[ERROR] Please provide a search query or --doi.", file=sys.stderr)
            sys.exit(1)
        download_by_search(query=query, count=count, year=args.year)


if __name__ == "__main__":
    main()
