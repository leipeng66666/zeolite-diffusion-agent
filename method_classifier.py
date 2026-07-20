# -*- coding: utf-8 -*-
"""Classify experimental methods in consolidated_results3.csv into Table 1 categories
from Kärger et al. "Diffusion in nanoporous materials".

Produces:
  1. method_mapping.csv — unique method → method_type + method_category
  2. Adds 'method_type' and 'method_category' columns to the source CSV
"""

import pandas as pd, os, time, logging, argparse
from pathlib import Path
from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("method_classifier")

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
    base_url="https://api.deepseek.com"
)

PROMPT_DIR = Path(__file__).with_name("prompts")
METHOD_PROMPT = (PROMPT_DIR / "method.txt").read_text(encoding="utf-8-sig").strip()

BATCH_SIZE = 10  # methods per LLM call
MAPPING_COLS = ["experimental_method", "method_type", "method_category"]


def _call_llm(methods: list[str]) -> list[str]:
    """Send a batch of methods to LLM, return classified lines."""
    text = "\n".join(methods)
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": f"{METHOD_PROMPT}\n\n{text}"}],
        temperature=0
    )
    return [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]


def _parse_classification(line: str) -> tuple[str, str, str]:
    """Parse 'method → Computational | MD' into (method, method_type, method_category)."""
    if "→" not in line:
        return line, "", ""
    method, rest = line.split("→", 1)
    method = method.strip()
    rest = rest.strip()
    # Split type and category: "Experimental | Uptake/Release" or "Computational | MD"
    if "|" in rest:
        mtype, category = rest.split("|", 1)
        return method, mtype.strip(), category.strip()
    else:
        # Fallback for old format "Experimental: Uptake/Release"
        if ":" in rest:
            mtype, category = rest.split(":", 1)
            return method, mtype.strip(), category.strip()
        return method, rest, ""


def classify_methods(csv_path: str) -> None:
    """Classify all unique experimental_method values and update the CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # --- Extract unique methods ---
    methods = df["experimental_method"].dropna().unique().tolist()
    logger.info(f"Found {len(methods)} unique experimental methods")

    # --- Checkpoint: load existing mapping ---
    mapping_path = Path(csv_path).parent / "method_mapping.csv"
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path, encoding="utf-8-sig")
        # Support both old (2-col) and new (3-col) mapping files
        for col in MAPPING_COLS:
            if col not in mapping_df.columns:
                mapping_df[col] = None
        done = set(mapping_df["experimental_method"].dropna().tolist())
    else:
        mapping_df = pd.DataFrame(columns=MAPPING_COLS)
        done = set()

    pending = [m for m in methods if str(m) not in done]
    logger.info(f"{len(done)} already classified, {len(pending)} pending")

    if not pending:
        logger.info("All methods already classified. Applying mapping to CSV...")
    else:
        # --- Batch classify ---
        new_rows = []
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            logger.info(f"Classifying batch {i//BATCH_SIZE + 1}/{(len(pending)-1)//BATCH_SIZE + 1}: {len(batch)} methods")
            try:
                lines = _call_llm(batch)
            except Exception as e:
                logger.error(f"LLM call failed: {e}, retrying in 5s...")
                time.sleep(5)
                try:
                    lines = _call_llm(batch)
                except Exception as e2:
                    logger.error(f"Retry also failed: {e2}, skipping batch")
                    continue

            for line in lines:
                method, mtype, category = _parse_classification(line)
                if method and mtype:
                    new_rows.append({
                        "experimental_method": method,
                        "method_type": mtype,
                        "method_category": category
                    })
                    logger.info(f"  {method[:70]} → {mtype} | {category}")
                else:
                    logger.warning(f"  Could not parse: {line[:100]}")

            # Save incrementally
            if new_rows:
                batch_df = pd.DataFrame(new_rows)
                mapping_df = pd.concat([mapping_df, batch_df], ignore_index=True)
                mapping_df = mapping_df[MAPPING_COLS]
                mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")
                new_rows = []
            time.sleep(1)

    # --- Apply mapping to main CSV ---
    mapping_df = pd.read_csv(mapping_path, encoding="utf-8-sig")
    for col in MAPPING_COLS:
        if col not in mapping_df.columns:
            mapping_df[col] = None
    type_lookup = dict(zip(mapping_df["experimental_method"], mapping_df["method_type"]))
    cat_lookup = dict(zip(mapping_df["experimental_method"], mapping_df["method_category"]))

    df["method_type"] = df["experimental_method"].map(type_lookup)
    df["method_category"] = df["experimental_method"].map(cat_lookup)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] method_type + method_category columns written to {csv_path}")

    # Summary statistics
    logger.info(f"\nBy type:\n{df['method_type'].value_counts().to_string()}")
    logger.info(f"\nBy category:\n{df['method_category'].value_counts().to_string()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Classify experimental methods into Table 1 categories")
    parser.add_argument("--input", required=True, help="Path to consolidated CSV file")
    args = parser.parse_args()
    classify_methods(args.input)
