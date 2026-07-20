# Zeolite Diffusion Data Extraction Agent

A two-stage LLM-based agent pipeline for extracting structured diffusion coefficient data from zeolite research papers (XML, HTML, Markdown).

## Overview

This system processes scientific papers about zeolite diffusion and extracts structured data using a **two-stage agent chain**:

1. **Stage 1 — Value Extraction** ([`value.py`](value.py)): Scans the full paper text to find and extract all numerical diffusion coefficient values. Uses paragraph-level splitting for thorough coverage.

2. **Stage 2 — Context Extraction** ([`Multidimensional-Data.py`](Multidimensional-Data.py)): For each diffusion value found in Stage 1, uses the full paper text to extract detailed contextual information (guest molecule, zeolite name, temperature, pressure, concentration, experimental method, etc.) with rigorous validation.

The two stages are orchestrated by [`agent.py`](agent.py), which also handles:
- XML → TXT conversion via [`xml-clean.py`](xml-clean.py) (parses Springer/Nature XML with MathML→LaTeX)
- HTML → TXT conversion via [`html-clean.py`](html-clean.py) (includes LLM-powered table reconstruction)
- Markdown pre-cleaning via [`md_clean.py`](md_clean.py) (PaddleOCR output cleanup)
- Checkpoint resume (skip already-processed papers)
- Post-processing pipeline (unit conversion, normalization, method classification)

### Why two stages?

A single-pass approach (sending the entire paper to the LLM and asking it to find both values and context) suffers from missed values and hallucinated context. The two-stage approach:

- Stage 1 focuses the LLM exclusively on **finding numbers** — a simpler task with higher recall.
- Stage 2 provides those numbers as anchors, asking the LLM to **validate and contextualize** — reducing hallucination because the model is filling in blanks rather than inventing from scratch.

## Extracted Data Fields

Each record contains up to 20 fields:

| Field | Description |
|-------|-------------|
| `guest_molecule` | Guest molecule name (e.g., CO2, CH4, H2O) |
| `guest_composition` | Mixture composition ratio (null if single component) |
| `zeolite_name` | Full zeolite designation (e.g., ZSM-5-50, NaY-2.5) |
| `si_al_ratio` | Si/Al ratio (dimensionless) |
| `modified_ion` | Modifying ion or metal (e.g., Na+, Fe, Pt) |
| `loading_value` / `loading_unit` | Ion/metal loading amount and unit |
| `diffusion_coefficient_value` / `diffusion_coefficient_unit` | D value in scientific notation |
| `temperature_value` / `temperature_unit` | Measurement temperature |
| `concentration_value` / `concentration_unit` | Fluid-phase concentration |
| `adsorption_loading_value` / `adsorption_loading_unit` | Adsorbate loading in zeolite pores |
| `pressure_value` / `pressure_unit` | Measurement pressure |
| `experimental_method` | Technique (PFG-NMR, ZLC, MD, QENS, etc.) |
| `distinguishing_variable` | Paper-specific differentiating variable |

## Project Structure

```
.
├── agent.py                      # Main orchestrator (two-stage agent pipeline)
├── value.py                      # Stage 1: diffusion value extraction
├── Multidimensional-Data.py      # Stage 2: context extraction & validation
├── xml-clean.py                  # XML → TXT converter
├── html-clean.py                 # HTML → TXT converter
├── md_clean.py                   # Markdown pre-cleaner
├── unit_converter.py             # Post-processing: unit conversion
├── unique_extractor.py           # Post-processing: unique value extraction
├── normalizer.py                 # Post-processing: name normalization
├── mapper.py                     # Post-processing: normalized name mapping
├── method_classifier.py          # Post-processing: method classification
├── cleaned_output.py             # Post-processing: final CSV generation
├── requirements.txt              # Python dependencies
├── prompts/                      # LLM prompt templates for post-processing
│   ├── unit.txt                  #   Diffusion unit conversion prompt
│   ├── temperature_unit.txt      #   Temperature unit conversion prompt
│   ├── concentration_unit.txt    #   Concentration unit conversion prompt
│   ├── pressure_unit.txt         #   Pressure unit conversion prompt
│   ├── temperature.txt           #   Temperature normalization prompt
│   ├── zeolite_name.txt          #   Zeolite name normalization prompt
│   ├── guest_molecule.txt        #   Guest molecule normalization prompt
│   └── method.txt                #   Method classification prompt
├── *_conversion_rules.csv        # Checkpoint files for unit conversions
├── method_mapping.csv            # Method type classification mapping
├── Document/                     # Input: papers (XML/TXT/MD) + per-paper outputs
│   └── (sample papers included)
├── Markdown/                     # Input: PaddleOCR-generated MD files
└── logs/                         # Runtime logs
```

## Quick Start

### Prerequisites

- Python 3.10+ (recommended 3.12)
- A [DeepSeek API key](https://platform.deepseek.com/) (or any OpenAI-compatible endpoint)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd xml-agent

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
# Windows:
set DEEPSEEK_API_KEY=your-key-here
# Linux/macOS:
export DEEPSEEK_API_KEY=your-key-here
```

### Usage

#### Option 1: Full pipeline (recommended)

```bash
python agent.py --mode full
```

You will be prompted to enter the workspace path. The workspace should contain:
- `Document/` — XML, TXT, MD, or HTML files to process
- `Markdown/` — (optional) PaddleOCR-generated MD files

The agent will:
1. Convert XML/HTML to TXT
2. Run Stage 1 (value extraction) on each paper
3. Run Stage 2 (context extraction) on each paper
4. Run post-processing (unit conversion, normalization, classification)
5. Output `consolidated_results3.csv` and `consolidated_results3_cleaned.csv`

Supports checkpoint resume — re-running will skip already-processed papers.

#### Option 2: Run individual stages

```bash
# Stage 1 only: extract diffusion values from TXT/MD files
python value.py --input Document --output Document

# Stage 2 only: extract context for pre-extracted values
python Multidimensional-Data.py --input Document --output consolidated_results3.csv
```

### Custom API Endpoint

To use a different OpenAI-compatible API (e.g., local vLLM, other providers):

```bash
# Windows:
set DEEPSEEK_BASE_URL=https://your-endpoint.com/v1
# Linux/macOS:
export DEEPSEEK_BASE_URL=https://your-endpoint.com/v1
```

### Input Format

Place papers in the `Document/` directory:

| Format | Example | Notes |
|--------|---------|-------|
| `.xml` | `S0009250909008896.xml` | Springer/Nature JATS XML. Requires companion `.txt` or will be auto-converted. |
| `.txt` | `S0009250909008896.txt` | Pre-extracted plain text. Skipped if XML source exists. |
| `.md` | `10.1002/aic.690420108.md` | PaddleOCR output. Auto-cleaned before extraction. |
| HTML folder | `S0009250909008896/` (containing `main.html`) | Springer/Nature HTML. Auto-converted to TXT. |

Sample papers are included in `Document/` for reference.

## How It Works

### Stage 1: Value Extraction (`value.py`)

The paper text is split into paragraphs. Each paragraph is sent to the LLM with a prompt focused exclusively on identifying diffusion coefficient values:

- Recognizes values in scientific notation, LaTeX, and plain formats
- Detects table multipliers (e.g., "D (10⁻⁹ m² s⁻¹)") and applies them
- Filters out non-diffusion values (rate constants, permeability, Arrhenius pre-exponential factors)
- Excludes values from cited references (not the paper's own data)

Output: a per-paper CSV with raw diffusion coefficient values.

### Stage 2: Context Extraction (`Multidimensional-Data.py`)

Each diffusion value from Stage 1 is presented to the LLM alongside the full paper text. The LLM must:

1. **Validate** each value against three criteria:
   - Is it a real zeolite diffusion coefficient?
   - Is it from *this* paper (not a reference)?
   - Is the value specific and explicit (not vague/approximate)?

2. **Extract context** for each validated value: scans the entire paper to find the guest molecule, zeolite name, temperature, pressure, etc. associated with each diffusion measurement.

3. **Output structured JSON** with all 20 fields per record.

Output: consolidated CSV with all extracted records.

### Post-Processing Pipeline

After extraction, several clean-up steps run automatically:

1. **Unit conversion** (`unit_converter.py`): Normalizes all units to standard forms (D → m²/s, T → K, concentration → mol/L, P → bar)
2. **Name normalization** (`normalizer.py` + `mapper.py`): Standardizes zeolite names and guest molecule names
3. **Method classification** (`method_classifier.py`): Categorizes experimental methods
4. **Final output** (`cleaned_output.py`): Generates the cleaned CSV

## Requirements

See [`requirements.txt`](requirements.txt) for Python package dependencies:

- `openai` ≥ 1.73.0 — API client for DeepSeek / OpenAI-compatible endpoints
- `pandas` ≥ 2.0.0 — Data manipulation
- `tqdm` ≥ 4.65.0 — Progress bars
- `watchdog` ≥ 4.0.0 — File system monitoring (watch mode)
- `lxml` ≥ 5.0.0 — XML/HTML parsing

## License

[Specify your license here]

## Citation

If you use this tool in your research, please cite:
[Add citation information]
