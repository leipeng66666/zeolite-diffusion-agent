# Wiley Online Library Paper Downloader

Downloads papers from Wiley Online Library via HTML scraping.

## Features

- Keyword search on Wiley Online Library
- Multiple download formats (PDF, HTML, XML)
- Batch download with configurable range
- Automatic file organization by format
- Built-in rate limiting

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive mode

```bash
python wiley_downloader.py
```

Follow the prompts to enter:
- Search keyword
- Download format (pdf / html / xml)
- Start position (e.g., 1)
- End position (e.g., 10)
- Output directory (default: downloads)

### Python API

```python
from wiley_downloader import WileyDownloader

downloader = WileyDownloader(output_dir="my_papers")
downloader.batch_download(
    keyword="zeolite diffusion",
    start=1,
    end=20,
    format_type="pdf"
)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `keyword` | Search keyword |
| `start` | Start position (1-indexed) |
| `end` | End position |
| `format_type` | `pdf`, `html`, or `xml` |
| `output_dir` | Save directory |

## Output Structure

```
downloads/
├── pdf/
│   ├── article1.pdf
│   └── article2.pdf
├── html/
│   ├── article1.html
│   └── article2.html
└── xml/
    ├── article1.xml
    └── article2.xml
```

## Notes

- Requires a stable network connection
- Some papers may require subscription access
- The downloader respects rate limits to avoid server overload
- Filenames are automatically sanitized and truncated to 100 characters
