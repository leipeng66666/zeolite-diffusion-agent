# Springer Link Paper Downloader

Downloads open-access Springer papers from Springer Link.

## Features

- Keyword search via Crossref API
- DOI-based single-paper download
- Multi-strategy PDF discovery (direct URL pattern, Springer Link scraping, content negotiation)
- Automatic deduplication via DOI hash
- Year filter support

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Search and download

```bash
python download.py "zeolite catalysis" --count 5
python download.py --search "porous materials" --year 2024 --count 10
```

### Download by DOI

```bash
python download.py --doi 10.1007/s10450-023-00412-3
```

### Command-line options

| Option | Description |
|--------|-------------|
| `query` | Search keyword (positional) |
| `--search`, `-s` | Search keyword (alternative) |
| `--doi`, `-d` | Download a specific paper by DOI |
| `--count`, `-c` | Number of papers (default: 5, max: 50) |
| `--year`, `-y` | Filter by publication year |
| `--output`, `-o` | Output directory (default: ./downloads) |

## API Configuration

Set the Crossref API endpoint before use:

```python
SEARCH_API = "https://api.crossref.org/works"
```

A Crossref API key ("polite pool") is optional but recommended for higher rate limits.
