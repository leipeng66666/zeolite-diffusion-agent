'''This code processes TXT files and uses a large model to extract structured diffusion data as JSON,
then converts to CSV with defined schema.
Change Log:
Date        | Author              | Description
------------|---------------------|---------------------------------------------
2025/05/15  | lp                | Basic functionality implemented
2025/06/10  | modified           | Changed from XML to TXT processing
2026/05/28  | modified           | Changed to JSON schema extraction with expanded fields'''
import re
import os
import json
import glob
from openai import OpenAI
import pandas as pd
from time import time, sleep
import traceback
import argparse

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
                base_url="https://api.deepseek.com")


def prompt(Q, typ):
    """Send prompt to LLM and get response"""
    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=Q,
                temperature=0,
                frequency_penalty=0,
                presence_penalty=0
            )
            break
        except Exception as e:
            print("An error occurred:", e)
            err_str = str(e)
            if 'data_inspection_failed' in err_str or 'inappropriate content' in err_str:
                print("Content moderation blocked, skipping this file")
                return Q, "", 0, 0
            elif 'Please reduce the length of the messages' in err_str:
                print('TRUNCATING')
                if len(Q) > 3:
                    Q.pop(3)
                else:
                    Q.pop(1)
            elif 'per min' in err_str:
                print("Sleeping for 15 sec.")
                sleep(15)

    return Q, response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens


# JSON Schema definition (used in prompts)
# All numeric fields with units are split into _value and _unit for easier table import
JSON_SCHEMA = """
{
  "data": [
    {
      "guest_molecule": "guest molecule name (e.g., CO2, CH4, H2O)",
      "guest_composition": "composition or ratio of guest molecule mixture (e.g., 50:50, equimolar, 10%CO2/90%CH4; write null if single component)",
      "zeolite_name": "zeolite name — preserve full designation with suffixes (e.g., ZSM-5, VET6, NaY-2.5, MFI)",
      "si_al_ratio": "Si/Al ratio (e.g., 50, 100, 200, dimensionless)",
      "modified_ion": "type of modifying ion or metal (e.g., Na+, K+, Ca2+, Mg2+, H+, Ag+, Fe, Cu, Pt, Pd; use None if not modified)",
      "loading_value": "loading amount of the modifying ion or metal (e.g., 5, 10, 2.5)",
      "loading_unit": "unit of the modifying ion or metal loading (e.g., wt%, %, g/g, mol%)",
      "diffusion_coefficient_value": "diffusion coefficient value in scientific notation (e.g., 1.5e-10)",
      "diffusion_coefficient_unit": "diffusion coefficient unit (e.g., m2/s, cm2/s)",
      "temperature_value": "measurement temperature value (e.g., 298, 25, 273)",
      "temperature_unit": "temperature unit (e.g., K, °C)",
      "concentration_value": "fluid-phase concentration value (e.g., 0.1, 1.0, 10)",
      "concentration_unit": "fluid-phase concentration unit (e.g., mol/L, mmol/L, mg/L, vol%, ppm)",
      "adsorption_loading_value": "gas adsorption loading in the zeolite (e.g., 16, 4, 0.17)",
      "adsorption_loading_unit": "gas adsorption loading unit (e.g., molecules/unit cell, mol/kg, wt%, cm3/g)",
      "pressure_value": "pressure value (e.g., 1, 101.325, 0.15)",
      "pressure_unit": "pressure unit (e.g., bar, kPa, atm, MPa)",
      "experimental_method": "experimental method used to measure the diffusion coefficient (e.g., PFG-NMR, Molecular Dynamics, Adsorption Kinetics, Membrane Permeation, Frequency Response, Chromatography, ZLC, QENS)",
      "distinguishing_variable": "paper-specific variable(s) beyond the standard fields above that distinguish different diffusion coefficients within the same paper. Format as 'name: value' pairs separated by semicolons. Examples: 'direction: radial', 'direction: axial', 'mode: adsorption', 'mode: desorption', 'membrane_type: hollow fiber', 'crystal_size: 33.3 μm', 'membrane_thickness: 5 μm', 'coating: PDMS', 'synthesis: seeded growth', 'modifier: none; direction: z'. Only include variables that meaningfully differentiate data points within THIS paper — if all records share the same condition, leave null."
    }
  ]
}
"""


def extract_diffusion_data(text, diffusion_values, filename):
    """Extract diffusion coefficient data from text using LLM, return structured JSON"""
    system_prompt = """You are a scientific research assistant specialized in chemistry and chemical engineering.
Your task is to validate, filter, and complete diffusion coefficient data extracted from zeolite research papers.

WORKFLOW - You MUST follow these steps in order:

STEP 1 — VALIDATION: For each given diffusion coefficient value, check ALL of the following criteria. If ANY criterion fails, DISCARD the value (do not include it in the output).

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

STEP 2 — COMPLETION: For each value that passed validation, extract all available context fields from the text. If a field is not mentioned anywhere in the paper, write null.
  - IMPORTANT: Extract data for EVERY zeolite type and every distinct condition reported in the paper. Do NOT focus only on the primary or most-mentioned zeolite — if the paper studies both VET and TON, both must appear in the output. If the paper studies the same zeolite under different loadings or cell sizes (e.g., VET6, VET12-double, VET32), each distinct configuration must have its own record.
  - Scan the ENTIRE paper for context. Context fields (zeolite name, temperature, guest molecule, etc.) may be mentioned in different sections than the diffusion coefficient value itself (e.g., the Experimental section, the abstract, figure/table captions).
  - If the paper studies multiple zeolites or conditions, match each diffusion value to its CORRECT corresponding context.
  - All numeric fields with a unit MUST be split into separate _value and _unit fields.

FIELD DESCRIPTIONS:
- guest_molecule: the molecule whose diffusion coefficient is measured (e.g., CO2, CH4, N2, H2O, xylene). Extract the exact name or formula as reported. If the guest is a mixture, list all components separated by commas or slashes (e.g., "CO2, CH4" or "CO2/CH4").
- guest_composition: the composition or mixing ratio of the guest mixture components (e.g., "50:50", "equimolar", "10%CO2/90%CH4", "25/75", "1:1"). Write null if the guest is a single component (not a mixture). Extract the ratio as reported in the paper — preserve the original format (molar ratio, volume ratio, mass ratio, etc.).
- zeolite_name: extract the FULL zeolite designation exactly as reported in the paper. Preserve ALL suffixes, numbers, and modifiers that distinguish different samples or conditions (e.g., "VET6", "VET12-double", "VET32", "ZSM-5-50", "NaY-2.5"). Do NOT simplify or truncate — "VET6" must stay "VET6", not become just "VET". If only the IZA code is given without modifiers, use that alone. Do NOT extract MOFs, COFs, or other porous materials.
- si_al_ratio: the Si/Al ratio of the zeolite (e.g., 50, 100). Dimensionless. Write null if not reported.
- modified_ion: the type of ion or metal used for zeolite modification (e.g., Na+, K+, Ca2+, Mg2+, H+, Ag+, Fe, Cu, Pt, Pd). Write null if the zeolite is not modified. Only extract when the ion/metal was intentionally introduced as a dopant/modifier — do NOT extract framework charge-balancing cations native to as-synthesized zeolites.
- loading_value: the loading amount (content) of the modifying ion or metal (e.g., 5, 10, 2.5, 0.5). This is the amount/concentration of the modifying species from `modified_ion`. Write null if no ion/metal modification. CRITICAL: this is ONLY for intentional dopant/ion-exchange loading — do NOT use this for guest molecule (adsorbate) loading, water density, "water loading", adsorbate uptake, or any measure of how much guest is inside the pores. Those belong in `adsorption_loading`.
- loading_unit: unit of the modifying ion or metal loading (e.g., wt%, %, g/g, mg/g, mol%). Write null if no ion/metal modification. CRITICAL: g/cm³ is NOT a valid ion loading unit — it indicates adsorbate density, which belongs in `adsorption_loading`.
- diffusion_coefficient_value: the validated numerical diffusion coefficient value in scientific notation.
- diffusion_coefficient_unit: unit of the diffusion coefficient (e.g., m2/s, cm2/s). Keep exactly as reported.
- temperature_value: the measurement temperature value corresponding to the diffusion coefficient (e.g., 298, 25). If a temperature range is reported, keep the range format (e.g., 273-323). If a single temperature applies to the entire experimental series, apply it to all data points. Never fill in material preparation temperatures (calcination, activation, etc.). If the paper says "room temperature" or "ambient temperature" without a specific value, fill in 298.
- temperature_unit: unit of the temperature (e.g., K, °C). Keep the original unit. When filling in room temperature, use K.
- concentration_value: numerical value of the FLUID-PHASE concentration at which the measurement was performed (e.g., 0.1, 10, 5). This is the concentration in the surrounding fluid (gas/liquid phase), NOT the amount adsorbed inside the zeolite. Write null if not reported.
- concentration_unit: unit of fluid-phase concentration (e.g., mol/L, mmol/L, mg/L, vol%, ppm). Write null if not reported.
- adsorption_loading_value: numerical value of the gas ADSORPTION LOADING inside the zeolite pores (e.g., 16, 4, 0.17, 0.29). This is the amount of adsorbate per unit mass/volume of zeolite or per unit cell. Write null if not reported.
- adsorption_loading_unit: unit of gas adsorption loading (e.g., molecules/unit cell, mol/kg, wt%, cm3/g, mg/g). Write null if not reported.
- pressure_value: numerical value of the measurement pressure or partial pressure (e.g., 1, 101.325). Priority: (1) value explicitly reported next to the diffusion data; (2) value stated for the whole experiment; (3) if the paper says "atmospheric pressure", "ambient pressure", or implies the experiment was conducted under normal atmosphere, fill in 1; (4) otherwise null.
- pressure_unit: unit of pressure (e.g., bar, kPa, MPa, atm, mmHg). Keep the original unit. When filling in atmospheric pressure, use bar.
- experimental_method: the experimental technique used (e.g., PFG-NMR, ZLC, gravimetric uptake, IR microscopy, membrane permeation, frequency response, TGA, Molecular Dynamics, Chromatography, QENS). Write null if not reported.
- distinguishing_variable: paper-specific variable(s) beyond the standard fields that distinguish different diffusion coefficients within the SAME paper. Format as 'name: value' pairs separated by semicolons (e.g., "direction: axial", "mode: desorption", "membrane_type: hollow fiber", "crystal_size: 33.3 μm", "coating: PDMS; thickness: 5 μm"). Only include variables that meaningfully differentiate data points from each other within THIS paper — if all records share the same condition across all records, write null. CRITICAL: look carefully at what makes each row of data different from other rows — is it diffusion direction? adsorption vs desorption? different membrane preparations? different crystal sizes? different synthesis methods? That distinguishing factor is what this field captures.

OUTPUT: Return ONLY a JSON object matching this schema. Discarded values are simply not included.
""" + JSON_SCHEMA

    user_prompt = f"""Execute the WORKFLOW described in the system prompt on the following paper.

Candidate diffusion coefficient values to validate: {diffusion_values}

For each value that passes validation, create one record in the JSON "data" array with all available context fields found in the text.
If no values pass validation, return {{"data": []}}.

Return ONLY the JSON object, no other text or explanation.

Paper text:
{text}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    _, summary, inp_tokens, out_tokens = prompt(messages, 'tab')
    return summary, inp_tokens, out_tokens


def parse_json_response(raw_response):
    """Parse JSON from LLM response, handling various formats"""
    if not raw_response or raw_response.strip() == '':
        return None

    text = raw_response.strip()

    # Try to extract JSON from markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    json_match = re.search(r'\{.*"data"\s*:\s*\[.*\]\s*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON array/object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"Warning: Could not parse JSON from LLM response")
    return None


# Expected CSV columns (matching JSON schema)
CSV_COLUMNS = [
    'filename',
    'doi',
    'guest_molecule',
    'guest_composition',
    'zeolite_name',
    'si_al_ratio',
    'modified_ion',
    'loading_value',
    'loading_unit',
    'diffusion_coefficient_value',
    'diffusion_coefficient_unit',
    'temperature_value',
    'temperature_unit',
    'concentration_value',
    'concentration_unit',
    'adsorption_loading_value',
    'adsorption_loading_unit',
    'pressure_value',
    'pressure_unit',
    'experimental_method',
    'distinguishing_variable'
]

FIELD_MAPPING = {
    'guest_molecule': 'guest_molecule',
    'guest_composition': 'guest_composition',
    'zeolite_name': 'zeolite_name',
    'si_al_ratio': 'si_al_ratio',
    'modified_ion': 'modified_ion',
    'loading_value': 'loading_value',
    'loading_unit': 'loading_unit',
    'diffusion_coefficient_value': 'diffusion_coefficient_value',
    'diffusion_coefficient_unit': 'diffusion_coefficient_unit',
    'temperature_value': 'temperature_value',
    'temperature_unit': 'temperature_unit',
    'concentration_value': 'concentration_value',
    'concentration_unit': 'concentration_unit',
    'adsorption_loading_value': 'adsorption_loading_value',
    'adsorption_loading_unit': 'adsorption_loading_unit',
    'pressure_value': 'pressure_value',
    'pressure_unit': 'pressure_unit',
    'experimental_method': 'experimental_method',
    'distinguishing_variable': 'distinguishing_variable'
}


def _upsert_csv(output_csv, filename, new_df):
    """Upsert records for a filename in the consolidated CSV: remove old rows then append new ones."""
    if os.path.exists(output_csv):
        try:
            old = pd.read_csv(output_csv)
            old = old[old['filename'].astype(str) != str(filename)]
            # Ensure all CSV_COLUMNS exist (backward compat: older CSVs may lack newly added columns)
            for col in CSV_COLUMNS:
                if col not in old.columns:
                    old[col] = None
        except Exception:
            old = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        old = pd.DataFrame(columns=CSV_COLUMNS)

    # Ensure new_df has all CSV_COLUMNS too
    for col in CSV_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = None

    combined = pd.concat([old, new_df], ignore_index=True)
    combined = combined[CSV_COLUMNS]
    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")


def _move_to_no_data(txt_file):
    """Move TXT, CSV, JSON files for a paper with no valid data to a no_data folder at project root."""
    src_dir = os.path.dirname(txt_file)
    project_dir = os.path.dirname(src_dir)
    filename = os.path.splitext(os.path.basename(txt_file))[0]
    no_data_dir = os.path.join(project_dir, 'no_data')
    os.makedirs(no_data_dir, exist_ok=True)

    for ext in ('.txt', '.csv', '.json', '.xml', '.md'):
        src = os.path.join(src_dir, filename + ext)
        if os.path.exists(src):
            dst = os.path.join(no_data_dir, filename + ext)
            os.replace(src, dst)
            print(f"  Moved {filename}{ext} -> no_data/")


def _extract_doi(txt_file):
    """Extract DOI from the corresponding XML file.
    Falls back to searching file content, then constructing from filename pattern."""
    base = os.path.splitext(txt_file)[0]
    fname = os.path.basename(txt_file)
    # 1) Primary: extract from XML <xocs:doi> tag
    xml_path = base + '.xml'
    if os.path.exists(xml_path):
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        m = re.search(r'<xocs:doi>(10\.\d{4,}/[^<]+)</xocs:doi>', content)
        if m:
            return m.group(1).strip()
    # 2) Search the text/markdown file itself for DOI patterns
    for ext in ('.txt', '.md'):
        src_path = base + ext
        if os.path.exists(src_path):
            with open(src_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Try "How to cite this article: ... https://doi.org/10.XXX/YYY" (common in Wiley MD files)
            m = re.search(r'https?://doi\.org/(10\.\d{4,}/[^\s]+)', content)
            if m:
                return m.group(1).strip().rstrip('.')
            # Try "doi:10.XXX/YYY" or "DOI: 10.XXX/YYY"
            m = re.search(r'(?:doi|DOI)\s*[:=]\s*(10\.\d{4,}/[^\s,;]+)', content)
            if m:
                return m.group(1).strip().rstrip('.')
            break  # only check one (prefer .txt over .md for content search)
    # 3) MD filename starts with DOI-like pattern (e.g., "10.1002-chem.201003596" → "10.1002/chem.201003596")
    if fname.startswith('10.') and not fname.startswith('S'):
        # Replace first '-' with '/'
        doi = re.sub(r'^(10\.\d{4,})-', r'\1/', os.path.splitext(fname)[0])
        if doi != fname:
            return doi
    # 4) Wiley journal abbreviation pattern (e.g., "adma.19950070103" → "10.1002/adma.19950070103")
    m = re.match(r'^([a-z]+)\.(\d{4,})', fname)
    if m:
        return f'10.1002/{os.path.splitext(fname)[0]}'
    # 5) Fallback: construct from Elsevier PII filename (S-prefix → 10.1016/PII)
    if fname.startswith('S'):
        return '10.1016/' + os.path.splitext(fname)[0]
    return None


def process_txt_file_with_diffusion_values(txt_file, diffusion_values, output_csv):
    """Process a single TXT file with diffusion values and save extracted data to CSV"""
    try:
        start_time = time()
        filename = os.path.splitext(os.path.basename(txt_file))[0]

        with open(txt_file, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # Extract diffusion data using LLM (now returns JSON)
        summary, inp_tokens, out_tokens = extract_diffusion_data(text_content, diffusion_values, filename)

        # Save raw LLM JSON response alongside TXT file
        json_path = os.path.join(os.path.dirname(txt_file), filename + '.json')
        with open(json_path, 'w', encoding='utf-8') as jf:
            jf.write(summary)
        print(f"JSON saved to {json_path}")

        # Parse JSON from LLM response
        parsed = parse_json_response(summary)

        if parsed is None:
            print(f"Warning: No valid JSON in LLM response for {os.path.basename(txt_file)}")
            _upsert_csv(output_csv, filename, pd.DataFrame(columns=CSV_COLUMNS))
            _move_to_no_data(txt_file)
            return False

        if 'data' not in parsed or not parsed['data']:
            print(f"Warning: No valid records after validation for {os.path.basename(txt_file)} — removing old entries if any")
            _upsert_csv(output_csv, filename, pd.DataFrame(columns=CSV_COLUMNS))
            _move_to_no_data(txt_file)
            return False

        records = parsed['data']

        # Hard filter: drop records without a valid zeolite name (non-zeolite materials)
        _invalid_zeolite = {'none', '', 'null', '-', '--', 'n/a', 'na', 'unknown', 'not specified', 'no zeolite', '—', '–', 'nan', 'undefined', 'nil', 'none.', 'n.a.', 'n/a.'}
        _non_zeolite = {'mof', 'cof', 'zif', 'carbon', 'silica gel', 'alumina', 'activated carbon', 'carbon nanotube', 'cnt', 'graphene', 'mesoporous silica', 'mcm-41', 'sba-15', 'mcm', 'sba'}
        def _valid_zeolite(r):
            zn = r.get('zeolite_name')
            if not zn:
                return False
            zn = str(zn).strip().lower()
            if not zn.strip():
                return False
            if zn in _invalid_zeolite:
                return False
            if zn in _non_zeolite:
                return False
            return True
        records = [r for r in records if _valid_zeolite(r)]

        # Deduplicate: LLM sometimes outputs identical records
        seen = set()
        unique_records = []
        for r in records:
            key = json.dumps(r, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                unique_records.append(r)
        if len(unique_records) < len(records):
            print(f"  Removed {len(records) - len(unique_records)} duplicate record(s)")
        records = unique_records

        if not records:
            print(f"Warning: All records discarded — no zeolite name found for {os.path.basename(txt_file)}")
            _upsert_csv(output_csv, filename, pd.DataFrame(columns=CSV_COLUMNS))
            _move_to_no_data(txt_file)
            return False

        extracted_data = []
        doi = _extract_doi(txt_file)
        for record in records:
            row = {'filename': filename, 'doi': doi if doi else 'None'}
            for json_field, csv_col in FIELD_MAPPING.items():
                val = record.get(json_field, 'None')
                val_str = str(val) if val is not None else 'None'
                # Normalize: LLM sometimes outputs "C" or "c" instead of "°C"
                if csv_col == 'temperature_unit' and val_str.strip().upper() in ('C', '°C', 'DEG C', 'DEG C.', 'DEGREES C', 'CELSIUS'):
                    val_str = '°C'
                row[csv_col] = val_str
            extracted_data.append(row)

        if extracted_data:
            new_df = pd.DataFrame(extracted_data)
            new_df = new_df[CSV_COLUMNS]
            _upsert_csv(output_csv, filename, new_df)
        else:
            print(f"Warning: No data records extracted for {os.path.basename(txt_file)}")
            _upsert_csv(output_csv, filename, pd.DataFrame(columns=CSV_COLUMNS))
            return False

        elapsed_time = time() - start_time
        print(f"Successfully processed {os.path.basename(txt_file)} in {elapsed_time:.2f} seconds")
        return True

    except Exception as e:
        print(f"Error processing {os.path.basename(txt_file)}: {str(e)}")
        traceback.print_exc()
        return False


def batch_process_with_diffusion_values(input_folder, output_csv):
    """Process TXT/MD files paired with CSV files of the same name.
    Supports checkpoint resume: files already written to the consolidated CSV are automatically skipped."""
    done_filenames = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'filename' in existing_df.columns:
                done_filenames = set(existing_df['filename'].astype(str).tolist())
                print(f"Checkpoint resume: detected {len(done_filenames)} completed files")
        except Exception:
            pass

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    total_files = len(csv_files)
    processed_files = 0
    skip_files = 0

    print(f"Found {total_files} CSV files to process")

    for csv_file in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]

        if base_name in done_filenames:
            skip_files += 1
            continue

        txt_file = os.path.join(input_folder, f"{base_name}.txt")
        md_file = os.path.join(input_folder, f"{base_name}.md")

        if os.path.exists(txt_file):
            text_file = txt_file
        elif os.path.exists(md_file):
            text_file = md_file
        else:
            print(f"Warning: No corresponding TXT or MD file found for {base_name}")
            continue

        try:
            diffusion_df = pd.read_csv(csv_file)
            if 'diffusion_value' not in diffusion_df.columns:
                print(f"Error: CSV file {csv_file} is missing 'diffusion_value' column")
                continue

            diffusion_values = diffusion_df['diffusion_value'].tolist()
            print(f"Loaded {len(diffusion_values)} values from {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
            continue

        print(f"\nProcessing pair {processed_files + 1}/{total_files}:")
        print(f"CSV: {os.path.basename(csv_file)}")
        print(f"Text: {os.path.basename(text_file)}")

        if process_txt_file_with_diffusion_values(text_file, diffusion_values, output_csv):
            processed_files += 1

    print(f"\nProcessing complete. Successfully processed {processed_files}/{total_files} file pairs (skipped {skip_files} already done)")
    print(f"Extracted data saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    batch_process_with_diffusion_values(args.input, args.output)
