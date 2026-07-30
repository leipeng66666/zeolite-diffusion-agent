# Zeolite Diffusion Data Extraction

A two-module pipeline for acquiring scientific papers from major publishers and extracting structured diffusion coefficient data from zeolite research.

## Project Structure

```
├── literature-acquisition/       # Paper download tools
│   ├── wiley/                    #   Wiley Online Library downloader
│   ├── elsevier/                 #   Elsevier / ScienceDirect downloader
│   └── springer/                 #   Springer Link downloader
├── literature-extraction/        # Data extraction pipeline
│   ├── agent.py                  #   Main orchestrator
│   ├── value.py                  #   Stage 1: diffusion value extraction
│   ├── Multidimensional-Data.py  #   Stage 2: context extraction
│   ├── ...                       #   Post-processing modules
│   └── Document/                 #   Sample papers
├── LICENSE                       # MIT License
└── .gitignore
```

## Literature Acquisition

Tools for downloading open-access papers from three major publishers. Each downloader supports keyword search and DOI-based download, with publisher-specific PDF discovery logic.

| Module | Publisher | Method |
|--------|-----------|--------|
| [wiley/](literature-acquisition/wiley/) | Wiley Online Library | HTML scraping |
| [elsevier/](literature-acquisition/elsevier/) | Elsevier / ScienceDirect | Crossref API + scraping |
| [springer/](literature-acquisition/springer/) | Springer Link | Crossref API + scraping |

See each subdirectory for usage instructions.

## Literature Extraction

A two-stage LLM-based agent pipeline for extracting structured diffusion coefficient data from zeolite research papers (XML, HTML, Markdown).

### How It Works

1. **Stage 1 — Value Extraction** ([`value.py`](literature-extraction/value.py)): Scans the full paper text paragraph-by-paragraph to find all numerical diffusion coefficient values.

2. **Stage 2 — Context Extraction** ([`Multidimensional-Data.py`](literature-extraction/Multidimensional-Data.py)): For each value found, uses the full text to extract detailed context (guest molecule, zeolite name, temperature, pressure, etc.) with rigorous validation.

The two stages are orchestrated by [`agent.py`](literature-extraction/agent.py), followed by automated post-processing (unit conversion, normalization, method classification).

See the [extraction README](literature-extraction/README.md) for full documentation.

## Quick Start

```bash
git clone https://github.com/leipeng66666/zeolite-diffusion-agent.git
cd zeolite-diffusion-agent

# Install extraction dependencies
pip install -r literature-extraction/requirements.txt

# Set your DeepSeek API key
export DEEPSEEK_API_KEY=your-key-here

# Run the extraction pipeline
python literature-extraction/agent.py --mode full
```

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{zeolite_diffusion_extraction,
  author       = {leipeng66666},
  title        = {Zeolite Diffusion Data Extraction},
  url          = {https://github.com/leipeng66666/zeolite-diffusion-agent},
  year         = {2026},
  note         = {Email: 1072534051@qq.com},
}
```
