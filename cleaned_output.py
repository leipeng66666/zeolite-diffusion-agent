# -*- coding: utf-8 -*-
"""Generate a cleaned CSV from the fully post-processed intermediate CSV.

Rules:
- Keep original zeolite_name AS-IS, AND include std_zeolite_name
- For columns that have been converted/normalized: use the converted version
- For columns that haven't been modified: keep original values
- Output to a new file (does NOT modify the input)
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger("cleaned_output")

# Final column order for cleaned CSV
CLEANED_COLUMNS = [
    "filename",
    "doi",
    "guest_molecule",            # → std_guest_molecule (fallback original)
    "guest_composition",
    "zeolite_name",              # ORIGINAL, kept as-is
    "std_zeolite_name",          # from normalization
    "si_al_ratio",
    "modified_ion",
    "loading_value",
    "loading_unit",
    "diffusion_coefficient_value",   # → converted_value
    "diffusion_coefficient_unit",    # → converted_unit
    "temperature_value",             # → temperature_K_value
    "temperature_unit",              # → temperature_K_unit
    "concentration_value",           # → concentration_converted_value
    "concentration_unit",            # → concentration_converted_unit
    "adsorption_loading_value",
    "adsorption_loading_unit",
    "pressure_value",                # → pressure_converted_value
    "pressure_unit",                 # → pressure_converted_unit
    "experimental_method",
    "distinguishing_variable",
    "method_type",
    "method_category",
]

# Mapping: cleaned column → (source column, fallback column)
# If source is None, keep original value (no transformation)
COLUMN_SOURCES = {
    "filename":                   (None, None),
    "doi":                        (None, None),
    "guest_molecule":             ("std_guest_molecule", "guest_molecule"),
    "guest_composition":          (None, None),
    "zeolite_name":               (None, None),           # ALWAYS keep raw original
    "std_zeolite_name":           ("std_zeolite_name", None),
    "si_al_ratio":                (None, None),
    "modified_ion":               (None, None),
    "loading_value":              (None, None),
    "loading_unit":               (None, None),
    "diffusion_coefficient_value": ("converted_value", "diffusion_coefficient_value"),
    "diffusion_coefficient_unit":  ("converted_unit", "diffusion_coefficient_unit"),
    "temperature_value":           ("temperature_K_value", "temperature_value"),
    "temperature_unit":            ("temperature_K_unit", "temperature_unit"),
    "concentration_value":         ("concentration_converted_value", "concentration_value"),
    "concentration_unit":          ("concentration_converted_unit", "concentration_unit"),
    "adsorption_loading_value":    (None, None),
    "adsorption_loading_unit":     (None, None),
    "pressure_value":              ("pressure_converted_value", "pressure_value"),
    "pressure_unit":               ("pressure_converted_unit", "pressure_unit"),
    "experimental_method":         (None, None),
    "distinguishing_variable":     (None, None),
    "method_type":                 ("method_type", None),
    "method_category":             ("method_category", None),
}


def generate_cleaned_csv(input_csv: str, output_csv: str = None) -> str:
    """Read the post-processed CSV and write a cleaned version.

    Args:
        input_csv: Path to the fully post-processed CSV (with all intermediate columns).
        output_csv: Output path. Default: <input_dir>/<input_name>_cleaned.csv

    Returns:
        Path to the generated cleaned CSV.
    """
    if output_csv is None:
        base = Path(input_csv)
        output_csv = str(base.parent / f"{base.stem}_cleaned.csv")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    logger.info(f"Read {len(df)} rows from {input_csv}")

    cleaned = pd.DataFrame()
    for col in CLEANED_COLUMNS:
        src, fallback = COLUMN_SOURCES.get(col, (None, None))
        if src is None:
            # No transformation — keep original
            cleaned[col] = df[col] if col in df.columns else None
        elif src in df.columns:
            # Use transformed value, with fallback to original
            series = df[src].copy()
            if fallback and fallback in df.columns:
                # Replace empty/NaN transformed values with original
                mask = series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.strip() == "nan")
                series[mask] = df.loc[mask, fallback]
            cleaned[col] = series
        elif fallback and fallback in df.columns:
            cleaned[col] = df[fallback]
        else:
            cleaned[col] = None

    cleaned.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] Cleaned CSV written: {output_csv} ({len(cleaned)} rows, {len(cleaned.columns)} columns)")
    return output_csv
