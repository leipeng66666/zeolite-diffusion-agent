import re
import os
import glob
import traceback
import pandas as pd
from openai import OpenAI
from time import sleep
import argparse

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
    base_url="https://api.deepseek.com"
)

def normalize_diffusion_value(raw):
    """
    Extract diffusion coefficient values from LaTeX or special formats into pure numeric format.
    Supported formats:
      - LaTeX:  $8\\,.\ ,31\\!\\times\\!10^{-6}...$  ->  8.31e-6
      - Unicode: 1.1e-11 or 1.1e-11  ->  1.1e-11
      - Standard:  8.31e-6  ->  8.31e-6
      - With error bar: 8.1±1.2 -> 8.1
    Returns None if unrecognized (preserved for original filter logic).
    """
    # Remove $...$ outer LaTeX frame
    s = re.sub(r'\$', '', raw)
    # Strip error/uncertainty notation: "8.1±1.2", "8.1+/-1.2", "8.1(2)" -> "8.1"
    s = re.sub(r'[±].*$', '', s)
    s = re.sub(r'\+/-.*$', '', s)
    s = re.sub(r'\([0-9]+\)', '', s)
    # Restore LaTeX space+decimal concatenation notation (e.g., 8\ ,.\,31 -> 8.31)
    s = re.sub(r'(\d)\\[,!;: ]*\.\\[,!;: ]*(\d)', r'\1.\2', s)
    # Remove LaTeX spacing commands (\ , \! \; \ )
    s = re.sub(r'\\[,!;: ]', '', s)
    # Normalize unicode minus (U+2212) to ASCII hyphen-minus
    s = s.replace('−', '-')
    # Replace \times or \cdot or Unicode × with *
    s = re.sub(r'\\times|\\cdot|×', '*', s)
    # Unicode middle dot also treated as multiplication
    s = s.replace('·', '*')
    # Remove font commands like \mathfrak{x} \pmb{x} \mathbf{x} (keep content)
    s = re.sub(r'\\(?:mathfrak|pmb|mathbf|mathsf|mathrm|mathtt|vec)\{([^{}]*)\}', r'\1', s)
    # Remove remaining LaTeX commands
    s = re.sub(r'\\[a-zA-Z]+\*?(?:\{[^{}]*\})*', '', s)
    # Target pattern: coefficient * 10^{exponent} or 10^exponent (with or without ^)
    # Case A: 1.1*10^{-11} or 1.1*10^-11
    m = re.search(
        r'([0-9]+(?:[.,][0-9]+)?)\s*\*\s*10\s*\^\s*\{?\s*([+-]?[0-9]+)\s*\}?',
        s
    )
    if m:
        coeff = m.group(1).replace(',', '.')
        exp = m.group(2)
        return f"{coeff}e{exp}"
    # Case B: 1.1*10-11 (no ^, middle-dot format)
    m_b = re.search(
        r'([0-9]+(?:[.,][0-9]+)?)\s*\*\s*10([+-][0-9]+)(?![0-9])',
        s
    )
    if m_b:
        coeff = m_b.group(1).replace(',', '.')
        exp = m_b.group(2)
        return f"{coeff}e{exp}"
    # Already in standard scientific notation, return directly
    m2 = re.search(r'([0-9]+(?:[.,][0-9]+)?[eE][+-]?[0-9]+)', s)
    if m2:
        return m2.group(1).replace(',', '.')
    # Plain number
    m3 = re.search(r'([0-9]+(?:[.,][0-9]+)?)', s)
    if m3:
        # Only convert if original string actually contains units/LaTeX/error bar
        if re.search(r'[a-zA-Z\\${}^_·×±]', raw):
            return m3.group(1).replace(',', '.')
    return None


def extract_diffusion_values(text):
    """Core extraction function, unchanged"""
    messages = [
        {"role": "user",
         "content": f'''STEP 1: Extract ALL numerical values explicitly identified as diffusion coefficients from the following text. CRITICAL: For TABLES, you MUST extract EVERY diffusion coefficient value from EVERY row and EVERY column — do not skip any row, do not sample, do not summarize. Each individual value goes on its own line.

If the table header indicates a multiplier (e.g., "D (10⁻⁹ m² s⁻¹)" or "D × 10⁹"), apply that multiplier to each value: a table entry "8.1±1.2" in a column labeled "10⁻⁹ m² s⁻¹" becomes "8.1e-9". Drop the ± error bar and output only the central value.

Focus specifically on values that are:

        Expressed in units characteristic of diffusion coefficients (e.g., m2/s, cm2/s)
        Clearly associated with molecular movement through zeolite pores
        Data origin must be from this paper as one of the following:
        Experimental results
        Measurement data
        Simulation data
        Calculated data
        Ignore any numbers that:

        Appear near but aren't explicitly identified as diffusion coefficients
        Represent other properties such as adsorption rate, activation energy, pore size, or rate constant
        Are Arrhenius pre-exponential factors (D₀, D0, A, D∞) — these are fit parameters from D = D₀·exp(-Ea/RT), NOT actual diffusion coefficients. Even though they share units (m²/s), they are extrapolated reference values, not measured/calculated D values at any real temperature.
        Are presented as ranges without clear diffusion coefficient attribution each on a separate line.
        Vague numerical values where the NUMBER ITSELF is imprecise — only the order of magnitude is given without specific digits (e.g., "~10⁻¹⁶", "approximately 10⁻¹⁰", "roughly on the order of 1e-9", "about 10⁻⁸ m²/s"). IMPORTANT: Do NOT discard values just because the authors describe the MEASUREMENT METHOD as "rough estimate", "very approximate", or "only rough estimates" — if the table or text gives a specific number (like 0.1, 1.5, 4.5), that number is precise and should be extracted.
Are described by the paper itself as inherently unreliable — the authors explicitly state the values are "very approximate estimates" AND the uncertainty is so large that it "precludes a valid estimate" or the values cannot be meaningfully quantified (this goes beyond mere "rough estimate" language to indicate the numbers are essentially qualitative).
Values from referenced literature: only exclude data when the text explicitly contains phrases such as:
"reported in ..."
"previous studies ..."
"compared with [ref]"
"according to [author/year]"
"from literature [ref]"
"cited from ..."
or any other clear indication that the value originates from an external reference.
Note: If the text indicates the authors' own simulation or calculation (e.g., "we calculated ...", "our simulation shows ..."), the value should be extracted even if it resembles literature values.
STEP 2 — VALIDATION: For each given diffusion coefficient value, check ALL of the following criteria. If ANY criterion fails, DISCARD the value (do not include it in the output).

  Criterion A — Is this a real zeolite diffusion coefficient?
    - The value must describe diffusion inside a ZEOLITE (microporous aluminosilicate, e.g., FAU, MFI, LTA, BEA, MOR, CHA, FER). ALL types of diffusion in zeolites are valid: intracrystalline diffusion, intercrystalline diffusion (diffusion in gaps/spaces between zeolite crystals, also known as bed diffusion or macroporous diffusion), Knudsen diffusivity (D_Kn), axial diffusion, radial diffusion, surface diffusion, micropore diffusion, macropore diffusion, effective diffusivity, self-diffusion, transport diffusion — all are needed. IMPORTANT: If the unit notation has a typo (e.g., m⁻² s⁻¹ instead of m² s⁻¹), still recognize it as a valid diffusion coefficient unit.
    - DISCARD ONLY if: (1) the value describes diffusion in MOFs, COFs, carbon molecular sieves, silica gel, alumina, or other non-zeolite materials, or (2) the value is a rate constant, adsorption coefficient, permeability, or Arrhenius pre-exponential factor (D₀/D0/A/D∞ from D = D₀·exp(-Ea/RT) — an extrapolated fit parameter, not an actual D at any real temperature) rather than a diffusion coefficient, or (3) the value is a purely assumed input parameter — a number the authors arbitrarily chose, not derived from any measurement/simulation/fitting. The ONLY indicator of this is explicit language like "D was assumed to be", "taking D =", "set D =", or a table explicitly titled "Model parameters" with no experimental basis, or (4) the value is a viscous diffusivity (D_L) — pressure-driven fluid flow in pores, which is not a zeolite micropore diffusion coefficient.

  Criterion B — Is this data from THIS paper (not a reference/citation)?
    - The value must be the authors' own experimental measurement, simulation result, or calculation.
    - DISCARD if: the text explicitly attributes the value to another paper (e.g., "reported by [ref]", "from literature", "according to [author]", "as measured by [ref]", "cited from", "previous studies showed D = X"). If the value appears only in a literature review section or comparison table citing other works, discard it.
    - KEEP if: the authors state they measured/calculated it themselves (e.g., "we measured", "our simulation yields", "the measured D is", "we obtained").

  Criterion C — Is the value specific and explicit?
    - DISCARD if: the value ITSELF is vague — only an order of magnitude without specific digits (e.g., "approximately 1e-10" with no precise number given, "~10⁻¹¹", "on the order of 1e-9"). Do NOT discard a value just because the authors describe the measurement as a "rough estimate" or "approximate" — if a specific number is provided (e.g., "axial diffusivity = 0.1" or "D ≈ 0.45 × 10⁻¹⁶"), the number IS precise and should be kept. HOWEVER, DISCARD if the paper explicitly states the estimate is so unreliable that meaningful quantification is impossible — look for language like "large uncertainty precludes a valid estimate", "the uncertainty is too large for meaningful quantification", or the values are explicitly described as "very approximate estimates" in a context where the paper acknowledges they cannot be trusted as quantitative data.
    - DISCARD if: the value is a range copied from another paper in a literature review section.
    - KEEP if: the value is reported with clear precision (e.g., "1.23e-10 m2/s", "D = 5.6 × 10^-11"). Values reported in figure captions or table notes are VALID and should be kept.

        If there are numerical values for the diffusion coefficient, for each extracted value, provide only the numerical value itself in standard scientific notation (e.g., 8.1e-9), each value on a separate line. If none exist, directly output "none".\n\n{text}'''}
    ]
    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            print(f"API Error: {err_str}")
            # Content moderation error, cannot retry, skip this paragraph
            if 'data_inspection_failed' in err_str or 'inappropriate content' in err_str:
                print("Content moderation blocked, skipping this paragraph")
                return "none"
            sleep(5)



def split_paragraphs(document):
    """Paragraph splitting function. Handles [Table_...] blocks and <html><body><table> blocks."""
    paragraphs = []
    current_paragraph = []
    in_table = False
    lines = document.split('\n')
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped_line = line.strip()
        # Detect [Table_ marker (TXT format)
        if stripped_line.startswith('[Table_'):
            if current_paragraph:
                paragraphs.append('\n'.join(current_paragraph))
                current_paragraph = []
            in_table = True
            current_paragraph.append(line)
            i += 1
        elif in_table:
            current_paragraph.append(line)
            if not stripped_line:
                if i + 1 < n and not lines[i + 1].strip():
                    paragraphs.append('\n'.join(current_paragraph))
                    current_paragraph = []
                    in_table = False
                    i += 2
                    continue
            i += 1
        else:
            if not stripped_line:
                if current_paragraph:
                    paragraphs.append('\n'.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line)
            i += 1
    if current_paragraph:
        paragraphs.append('\n'.join(current_paragraph))
    return paragraphs

def process_txt_strict(txt_path):
    """Process a single TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        blocks = split_paragraphs(content)
        results = []
        for block in blocks:
            if not block.strip():
                continue
            if block.startswith('[Table'):
                table_content = '\n'.join(block.split('\n')[1:]).strip()
                values = extract_diffusion_values(table_content)
            else:
                values = extract_diffusion_values(block)
            if values:
                for v in values.split('\n'):
                    if v.strip():
                        results.append({"diffusion_value": v.strip()})
        return results
    except Exception as e:
        print(f"Error processing file: {txt_path}\n{str(e)}")
        traceback.print_exc()
        return []


def _load_processing_log(log_path):
    """Load processing log: filename -> status (success/empty/error)."""
    if not os.path.exists(log_path):
        return {}
    try:
        df = pd.read_csv(log_path, encoding="utf-8-sig")
        if "filename" not in df.columns or "status" not in df.columns:
            return {}
        return dict(zip(df["filename"], df["status"]))
    except Exception:
        return {}

def _save_processing_log(log_path, log_dict):
    """Save processing log to CSV."""
    df = pd.DataFrame(
        [{"filename": k, "status": v} for k, v in log_dict.items()],
        columns=["filename", "status"]
    )
    df.to_csv(log_path, index=False, encoding="utf-8-sig")

def batch_process_strict(input_folder, output_folder):
    """Batch process all files (.txt and .md). Supports checkpoint resume: files already processed (success, empty, or error) are automatically skipped."""
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Load processing log (tracks ALL files: success, empty, error)
    log_path = os.path.join(output_folder, "processing_log.csv")
    proc_log = _load_processing_log(log_path)

    # Get all TXT and MD files
    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    md_files = glob.glob(os.path.join(input_folder, "*.md"))
    all_files = txt_files + md_files

    # Count completed and pending
    skip_csv = 0
    skip_log = 0
    pending = []
    for f in all_files:
        base_name = os.path.basename(f)
        # Check 1: CSV already exists (success from previous run)
        csv_name = base_name.replace('.txt', '.csv').replace('.md', '.csv')
        csv_path = os.path.join(output_folder, csv_name)
        if os.path.exists(csv_path):
            skip_csv += 1
            # Also update log if missing
            if base_name not in proc_log:
                proc_log[base_name] = "success"
            continue
        # Check 2: In processing log as empty or error (don't re-process)
        if base_name in proc_log:
            skip_log += 1
            continue
        pending.append(f)

    if skip_log > 0:
        _save_processing_log(log_path, proc_log)

    print(f"Found {len(txt_files)} TXT files, {len(md_files)} MD files, {len(all_files)} total")
    print(f"Checkpoint resume: {skip_csv} with CSV, {skip_log} in log (empty/error), {len(pending)} pending")

    # Processing stats
    success_count = 0
    empty_count = 0
    error_count = 0

    for txt_file in pending:
        base_name = os.path.basename(txt_file)
        try:
            # Process file
            results = process_txt_strict(txt_file)

            if results:
                # Generate output path
                csv_name = base_name.replace('.txt', '.csv').replace('.md', '.csv')
                csv_path = os.path.join(output_folder, csv_name)

                # Create DataFrame and save
                df = pd.DataFrame(results)
                if not df.empty:
                    # Convert LaTeX strings to pure numbers first, then filter
                    def _clean(v):
                        normed = normalize_diffusion_value(v)
                        return normed if normed is not None else v
                    df['diffusion_value'] = df['diffusion_value'].apply(_clean)
                    # Filter: keep pure numbers and scientific notation (e/E), exclude unit letters and none
                    def _is_valid(v):
                        v = str(v).strip()
                        if v.lower() == 'none' or v == '':
                            return False
                        return bool(re.fullmatch(r'[+-]?[0-9]+(?:[.,][0-9]+)?(?:[eE][+-]?[0-9]+)?', v))
                    df = df[df['diffusion_value'].apply(_is_valid)]

                if not df.empty:
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    proc_log[base_name] = "success"
                    success_count += 1
                    print(f"[OK] Successfully processed: {base_name} -> {csv_name}")
                else:
                    proc_log[base_name] = "empty"
                    empty_count += 1
                    print(f"[WARN] No valid data: {base_name}")
            else:
                proc_log[base_name] = "empty"
                empty_count += 1
                print(f"[WARN] No extraction results: {base_name}")

        except Exception as e:
            proc_log[base_name] = "error"
            error_count += 1
            print(f"[FAIL] Processing failed: {base_name} - {str(e)}")
            traceback.print_exc()

        # Save log after each file for crash-safety
        _save_processing_log(log_path, proc_log)

    # Final report
    print(f"\nProcessing complete!")
    print(f"Success: {success_count} | Empty (no data): {empty_count} | Error: {error_count}")
    print(f"Output directory: {output_folder}")
    print(f"Processing log: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    # Execute batch processing
    batch_process_strict(input_folder, output_folder)
