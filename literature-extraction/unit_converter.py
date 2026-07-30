# -*- coding: utf-8 -*-
"""Convert diffusion coefficient units to m²/s.

NEW APPROACH (no per-row LLM calls):
  1. Collect all unique units from the CSV
  2. LLM writes one conversion rule per unique unit (ONE API call total)
  3. Apply rules programmatically — value × factor, computed in Python

Produces:
  1. unit_conversion_rules.csv — unique unit → operation + factor (checkpoint)
  2. Updates 'converted_value' and 'converted_unit' columns in the source CSV
"""

import pandas as pd, re, logging, argparse, time, math
from pathlib import Path
from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("unit_converter")

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
    base_url="https://api.deepseek.com"
)

PROMPT_DIR = Path(__file__).with_name("prompts")
UNIT_PROMPT = (PROMPT_DIR / "unit.txt").read_text(encoding="utf-8-sig").strip()
TEMP_UNIT_PROMPT = (PROMPT_DIR / "temperature_unit.txt").read_text(encoding="utf-8-sig").strip()
CONC_UNIT_PROMPT = (PROMPT_DIR / "concentration_unit.txt").read_text(encoding="utf-8-sig").strip()
PRESS_UNIT_PROMPT = (PROMPT_DIR / "pressure_unit.txt").read_text(encoding="utf-8-sig").strip()

BATCH_SIZE = 30  # unique units per LLM call (far fewer than per-row batches)
RULES_COLS = ["original_unit", "operation", "factor"]


def _call_llm(units: list[str]) -> list[str]:
    """Send all unique units to LLM, return rule lines."""
    text = "\n".join(units)
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": f"{UNIT_PROMPT}\n\n{text}"}],
        temperature=0
    )
    return [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]


def _parse_rule(line: str) -> tuple[str, str, str]:
    """Parse '<unit> → <operation> | <factor>' into (unit, operation, factor)."""
    if "→" not in line:
        return "", "", ""
    unit, rest = line.split("→", 1)
    unit = unit.strip()
    rest = rest.strip()
    if "|" not in rest:
        return unit, rest, ""
    operation, factor = rest.split("|", 1)
    return unit, operation.strip(), factor.strip()


def _apply_rule(value, operation: str, factor_str: str, original_unit: str, target_unit: str = "m2/s"):
    """Apply a conversion rule to a single value.

    Args:
        value: the original numeric value
        operation: "multiply", "identity", "keep", etc.
        factor_str: the conversion factor string
        original_unit: the original unit string
        target_unit: the target unit for converted values (e.g. "m2/s", "mol/L", "bar")

    Returns (converted_value, converted_unit).
    """
    # --- keep: don't convert ---
    if operation == "keep":
        return str(value), str(original_unit)

    # --- identity: already in target unit, just normalize ---
    if operation == "identity":
        return str(value), target_unit

    # --- multiply: value × factor ---
    if operation == "multiply":
        try:
            factor = float(factor_str)
            numeric = float(value)
            result = numeric * factor
            # Format: use scientific notation for very small/large numbers
            if abs(result) < 1e-4 or abs(result) >= 1e6:
                return f"{result:.6e}", target_unit
            else:
                return f"{result:.6f}", target_unit
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert value '{value}' with factor {factor_str}")
            return str(value), str(original_unit)

    # --- unknown operation ---
    logger.warning(f"Unknown operation '{operation}' for unit '{original_unit}', keeping as-is")
    return str(value), str(original_unit)


def convert_unit(csv_path: str) -> None:
    """Convert all diffusion_coefficient_value to m²/s using per-unit conversion rules."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # ---------- Ensure target columns exist ----------
    for col in ["converted_value", "converted_unit"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["converted_value"] = df["converted_value"].astype("string")
    df["converted_unit"] = df["converted_unit"].astype("string")

    # ---------- Step 1: Collect unique units ----------
    all_units = df["diffusion_coefficient_unit"].dropna().unique().tolist()
    # Add "none" explicitly as a unit to classify (handles NaN rows)
    has_nan = df["diffusion_coefficient_unit"].isna().any()
    unique_units = sorted(set(str(u).strip() for u in all_units))
    if has_nan:
        unique_units.append("none")
    logger.info(f"Found {len(unique_units)} unique units")

    # ---------- Step 2: Load / build conversion rules (checkpoint) ----------
    rules_path = Path(csv_path).parent / "unit_conversion_rules.csv"
    if rules_path.exists():
        rules_df = pd.read_csv(rules_path, encoding="utf-8-sig")
        for col in RULES_COLS:
            if col not in rules_df.columns:
                rules_df[col] = None
        done = set(rules_df["original_unit"].dropna().tolist())
        logger.info(f"Loaded {len(done)} existing rules from checkpoint")
    else:
        rules_df = pd.DataFrame(columns=RULES_COLS)
        done = set()

    pending = [u for u in unique_units if u not in done]
    logger.info(f"{len(done)} already classified, {len(pending)} pending")

    if pending:
        # ---------- Step 3: LLM writes rules for pending units ----------
        new_rows = []
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            logger.info(f"Classifying batch {i//BATCH_SIZE + 1}/{(len(pending)-1)//BATCH_SIZE + 1}: {len(batch)} units")
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
                unit, operation, factor = _parse_rule(line)
                if unit and operation:
                    new_rows.append({
                        "original_unit": unit,
                        "operation": operation,
                        "factor": factor
                    })
                    logger.info(f"  {unit[:60]} → {operation} | {factor}")
                else:
                    logger.warning(f"  Could not parse: {line[:100]}")

            # Save incrementally
            if new_rows:
                batch_df = pd.DataFrame(new_rows)
                rules_df = pd.concat([rules_df, batch_df], ignore_index=True)
                rules_df = rules_df[RULES_COLS]
                rules_df.to_csv(rules_path, index=False, encoding="utf-8-sig")
                new_rows = []
            time.sleep(0.5)

    # ---------- Step 4: Apply rules programmatically ----------
    rules_df = pd.read_csv(rules_path, encoding="utf-8-sig")
    rule_map = {}
    for _, row in rules_df.iterrows():
        rule_map[str(row["original_unit"]).strip()] = (
            str(row["operation"]).strip(),
            str(row["factor"]).strip()
        )

    converted_count = 0
    kept_count = 0
    identity_count = 0

    for idx in df.index:
        raw_unit = df.at[idx, "diffusion_coefficient_unit"]
        raw_value = df.at[idx, "diffusion_coefficient_value"]

        # Normalize the unit key
        if pd.isna(raw_unit):
            unit_key = "none"
        else:
            unit_key = str(raw_unit).strip()

        if unit_key not in rule_map:
            # Fallback: try case-insensitive match
            match = None
            for k in rule_map:
                if k.lower() == unit_key.lower():
                    match = k
                    break
            if match:
                unit_key = match
            else:
                logger.warning(f"No rule for unit '{unit_key}', keeping as-is")
                df.at[idx, "converted_value"] = str(raw_value)
                df.at[idx, "converted_unit"] = unit_key
                kept_count += 1
                continue

        operation, factor = rule_map[unit_key]

        if pd.isna(raw_value):
            df.at[idx, "converted_value"] = str(raw_value)
            df.at[idx, "converted_unit"] = unit_key if operation == "keep" else "m2/s"
            continue

        converted_val, converted_unit = _apply_rule(raw_value, operation, factor, unit_key)

        if operation == "multiply":
            converted_count += 1
        elif operation == "identity":
            identity_count += 1
        else:
            kept_count += 1

        df.at[idx, "converted_value"] = converted_val
        df.at[idx, "converted_unit"] = converted_unit

    # ---------- Save ----------
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] Unit conversion complete: {converted_count} converted, {identity_count} already m2/s, {kept_count} kept as-is")
    logger.info(f"Results written to {csv_path}")


# ============================================================
#  Temperature conversion: all temperatures → Kelvin (K)
#  Same approach: collect unique units → LLM rules → apply
# ============================================================

TEMP_RULES_COLS = ["original_unit", "operation", "mul_factor", "add_factor"]


def _call_llm_temp(units: list[str]) -> list[str]:
    """Send all unique temperature units to LLM, return rule lines."""
    text = "\n".join(units)
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": f"{TEMP_UNIT_PROMPT}\n\n{text}"}],
        temperature=0
    )
    return [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]


def _parse_temp_rule(line: str) -> tuple[str, str, str, str]:
    """Parse '<unit> → <operation> | <factor>' into (unit, operation, mul_factor, add_factor)."""
    if "→" not in line:
        return "", "", "", ""
    unit, rest = line.split("→", 1)
    unit = unit.strip()
    rest = rest.strip()
    if "|" not in rest:
        return unit, rest, "", ""
    operation, factor_str = rest.split("|", 1)
    operation = operation.strip()
    factor_str = factor_str.strip()

    if operation == "identity":
        return unit, operation, "1", "0"
    elif operation == "add":
        return unit, operation, "1", factor_str
    elif operation == "affine":
        if "," in factor_str:
            mul, add = factor_str.split(",", 1)
            return unit, operation, mul.strip(), add.strip()
        else:
            return unit, operation, factor_str, "0"
    elif operation == "keep":
        return unit, operation, "", ""
    else:
        return unit, operation, factor_str, "0"


def _apply_temp_rule(value, operation: str, mul_factor_str: str, add_factor_str: str, original_unit: str):
    """Apply a temperature conversion rule. Returns (converted_value_K, "K")."""
    if operation == "keep":
        return str(value), str(original_unit)

    if operation == "identity":
        return str(value), "K"

    try:
        numeric = float(value)
    except (ValueError, TypeError):
        logger.warning(f"Cannot convert temperature value '{value}', keeping as-is")
        return str(value), str(original_unit)

    if operation == "add":
        try:
            offset = float(add_factor_str)
            result = numeric + offset
        except (ValueError, TypeError):
            logger.warning(f"Cannot add offset '{add_factor_str}' to value '{value}'")
            return str(value), str(original_unit)

    elif operation == "affine":
        try:
            mul = float(mul_factor_str)
            add = float(add_factor_str)
            result = numeric * mul + add
        except (ValueError, TypeError):
            logger.warning(f"Cannot apply affine conversion to '{value}'")
            return str(value), str(original_unit)

    else:
        logger.warning(f"Unknown temp operation '{operation}' for unit '{original_unit}'")
        return str(value), str(original_unit)

    # Format: round to integer Kelvin (no decimals)
    return str(round(result)), "K"


def convert_temperature_to_kelvin(csv_path: str) -> None:
    """Convert all temperature_value to Kelvin using per-unit conversion rules."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # ---------- Ensure target columns exist ----------
    for col in ["temperature_K_value", "temperature_K_unit"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["temperature_K_value"] = df["temperature_K_value"].astype("string")
    df["temperature_K_unit"] = df["temperature_K_unit"].astype("string")

    # ---------- Step 1: Collect unique temperature units ----------
    all_units = df["temperature_unit"].dropna().unique().tolist()
    has_nan = df["temperature_unit"].isna().any()
    unique_units = sorted(set(str(u).strip() for u in all_units))
    if has_nan:
        unique_units.append("none")
    logger.info(f"[Temp] Found {len(unique_units)} unique temperature units")

    # ---------- Step 2: Load / build conversion rules (checkpoint) ----------
    rules_path = Path(csv_path).parent / "temperature_conversion_rules.csv"
    if rules_path.exists():
        rules_df = pd.read_csv(rules_path, encoding="utf-8-sig")
        for col in TEMP_RULES_COLS:
            if col not in rules_df.columns:
                rules_df[col] = None
        done = set(rules_df["original_unit"].dropna().tolist())
        logger.info(f"[Temp] Loaded {len(done)} existing rules from checkpoint")
    else:
        rules_df = pd.DataFrame(columns=TEMP_RULES_COLS)
        done = set()

    pending = [u for u in unique_units if u not in done]
    logger.info(f"[Temp] {len(done)} already classified, {len(pending)} pending")

    if pending:
        # ---------- Step 3: LLM writes rules for pending units ----------
        new_rows = []
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            logger.info(f"[Temp] Classifying batch {i//BATCH_SIZE + 1}/{(len(pending)-1)//BATCH_SIZE + 1}: {len(batch)} units")
            try:
                lines = _call_llm_temp(batch)
            except Exception as e:
                logger.error(f"[Temp] LLM call failed: {e}, retrying in 5s...")
                time.sleep(5)
                try:
                    lines = _call_llm_temp(batch)
                except Exception as e2:
                    logger.error(f"[Temp] Retry also failed: {e2}, skipping batch")
                    continue

            for line in lines:
                unit, operation, mul_factor, add_factor = _parse_temp_rule(line)
                if unit and operation:
                    new_rows.append({
                        "original_unit": unit,
                        "operation": operation,
                        "mul_factor": mul_factor,
                        "add_factor": add_factor
                    })
                    logger.info(f"  [Temp] {unit[:60]} → {operation} | mul={mul_factor} add={add_factor}")
                else:
                    logger.warning(f"  [Temp] Could not parse: {line[:100]}")

            # Save incrementally
            if new_rows:
                batch_df = pd.DataFrame(new_rows)
                rules_df = pd.concat([rules_df, batch_df], ignore_index=True)
                rules_df = rules_df[TEMP_RULES_COLS]
                rules_df.to_csv(rules_path, index=False, encoding="utf-8-sig")
                new_rows = []
            time.sleep(0.5)

    # ---------- Step 4: Apply rules programmatically ----------
    rules_df = pd.read_csv(rules_path, encoding="utf-8-sig")
    rule_map = {}
    for _, row in rules_df.iterrows():
        rule_map[str(row["original_unit"]).strip()] = (
            str(row["operation"]).strip(),
            str(row["mul_factor"]).strip() if pd.notna(row["mul_factor"]) else "",
            str(row["add_factor"]).strip() if pd.notna(row["add_factor"]) else ""
        )

    converted_count = 0
    identity_count = 0
    kept_count = 0

    for idx in df.index:
        raw_unit = df.at[idx, "temperature_unit"]
        raw_value = df.at[idx, "temperature_value"]

        # Normalize the unit key
        if pd.isna(raw_unit):
            unit_key = "none"
        else:
            unit_key = str(raw_unit).strip()

        if unit_key not in rule_map:
            # Fallback: try case-insensitive match
            match = None
            for k in rule_map:
                if k.lower() == unit_key.lower():
                    match = k
                    break
            if match:
                unit_key = match
            else:
                logger.warning(f"[Temp] No rule for unit '{unit_key}', keeping as-is")
                df.at[idx, "temperature_K_value"] = str(raw_value) if pd.notna(raw_value) else ""
                df.at[idx, "temperature_K_unit"] = unit_key
                kept_count += 1
                continue

        operation, mul_factor, add_factor = rule_map[unit_key]

        if pd.isna(raw_value):
            df.at[idx, "temperature_K_value"] = ""
            df.at[idx, "temperature_K_unit"] = "K" if operation in ("identity", "add", "affine") else unit_key
            continue

        converted_val, converted_unit = _apply_temp_rule(raw_value, operation, mul_factor, add_factor, unit_key)

        if operation == "identity":
            identity_count += 1
        elif operation in ("add", "affine"):
            converted_count += 1
        else:
            kept_count += 1

        df.at[idx, "temperature_K_value"] = converted_val
        df.at[idx, "temperature_K_unit"] = converted_unit

    # ---------- Save ----------
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"[Temp] [OK] Temperature conversion complete: {converted_count} converted, {identity_count} already K, {kept_count} kept as-is")
    logger.info(f"[Temp] Results written to {csv_path}")


# ============================================================
#  Concentration conversion: all → mol/L
#  Uses same multiply/identity/keep pattern as diffusion coefficient
# ============================================================

def _call_llm_conc(units: list[str]) -> list[str]:
    """Send all unique concentration units to LLM, return rule lines."""
    text = "\n".join(units)
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": f"{CONC_UNIT_PROMPT}\n\n{text}"}],
        temperature=0
    )
    return [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]


def convert_concentration_to_mol_per_l(csv_path: str) -> None:
    """Convert all concentration_value to mol/L using per-unit conversion rules."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    for col in ["concentration_converted_value", "concentration_converted_unit"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["concentration_converted_value"] = df["concentration_converted_value"].astype("string")
    df["concentration_converted_unit"] = df["concentration_converted_unit"].astype("string")

    # Collect unique units
    all_units = df["concentration_unit"].dropna().unique().tolist()
    has_nan = df["concentration_unit"].isna().any()
    unique_units = sorted(set(str(u).strip() for u in all_units))
    if has_nan:
        unique_units.append("none")
    logger.info(f"[Conc] Found {len(unique_units)} unique concentration units")

    # Load / build rules
    rules_path = Path(csv_path).parent / "concentration_conversion_rules.csv"
    if rules_path.exists():
        rules_df = pd.read_csv(rules_path, encoding="utf-8-sig")
        for col in RULES_COLS:
            if col not in rules_df.columns:
                rules_df[col] = None
        done = set(rules_df["original_unit"].dropna().tolist())
        logger.info(f"[Conc] Loaded {len(done)} existing rules from checkpoint")
    else:
        rules_df = pd.DataFrame(columns=RULES_COLS)
        done = set()

    pending = [u for u in unique_units if u not in done]
    logger.info(f"[Conc] {len(done)} already classified, {len(pending)} pending")

    if pending:
        new_rows = []
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            logger.info(f"[Conc] Classifying batch {i//BATCH_SIZE + 1}/{(len(pending)-1)//BATCH_SIZE + 1}: {len(batch)} units")
            try:
                lines = _call_llm_conc(batch)
            except Exception as e:
                logger.error(f"[Conc] LLM call failed: {e}, retrying in 5s...")
                time.sleep(5)
                try:
                    lines = _call_llm_conc(batch)
                except Exception as e2:
                    logger.error(f"[Conc] Retry also failed: {e2}, skipping batch")
                    continue

            for line in lines:
                unit, operation, factor = _parse_rule(line)
                if unit and operation:
                    new_rows.append({"original_unit": unit, "operation": operation, "factor": factor})
                    logger.info(f"  [Conc] {unit[:60]} → {operation} | {factor}")
                else:
                    logger.warning(f"  [Conc] Could not parse: {line[:100]}")

            if new_rows:
                batch_df = pd.DataFrame(new_rows)
                rules_df = pd.concat([rules_df, batch_df], ignore_index=True)
                rules_df = rules_df[RULES_COLS]
                rules_df.to_csv(rules_path, index=False, encoding="utf-8-sig")
                new_rows = []
            time.sleep(0.5)

    # Apply rules
    rules_df = pd.read_csv(rules_path, encoding="utf-8-sig")
    rule_map = {}
    for _, row in rules_df.iterrows():
        rule_map[str(row["original_unit"]).strip()] = (str(row["operation"]).strip(), str(row["factor"]).strip())

    converted_count = 0; identity_count = 0; kept_count = 0
    for idx in df.index:
        raw_unit = df.at[idx, "concentration_unit"]
        raw_value = df.at[idx, "concentration_value"]
        if pd.isna(raw_unit):
            unit_key = "none"
        else:
            unit_key = str(raw_unit).strip()

        if unit_key not in rule_map:
            match = next((k for k in rule_map if k.lower() == unit_key.lower()), None)
            if match:
                unit_key = match
            else:
                logger.warning(f"[Conc] No rule for unit '{unit_key}', keeping as-is")
                df.at[idx, "concentration_converted_value"] = str(raw_value) if pd.notna(raw_value) else ""
                df.at[idx, "concentration_converted_unit"] = unit_key
                kept_count += 1
                continue

        operation, factor = rule_map[unit_key]
        if pd.isna(raw_value):
            df.at[idx, "concentration_converted_value"] = ""
            df.at[idx, "concentration_converted_unit"] = "mol/L" if operation in ("multiply", "identity") else unit_key
            continue

        converted_val, converted_unit = _apply_rule(raw_value, operation, factor, unit_key, target_unit="mol/L")
        if operation == "multiply":
            converted_count += 1
        elif operation == "identity":
            identity_count += 1
        else:
            kept_count += 1
        df.at[idx, "concentration_converted_value"] = converted_val
        df.at[idx, "concentration_converted_unit"] = converted_unit

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"[Conc] [OK] Concentration conversion complete: {converted_count} converted, {identity_count} already mol/L, {kept_count} kept as-is")


# ============================================================
#  Pressure conversion: all → bar
#  Uses same multiply/identity/keep pattern
# ============================================================

def _call_llm_press(units: list[str]) -> list[str]:
    """Send all unique pressure units to LLM, return rule lines."""
    text = "\n".join(units)
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": f"{PRESS_UNIT_PROMPT}\n\n{text}"}],
        temperature=0
    )
    return [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]


def convert_pressure_to_bar(csv_path: str) -> None:
    """Convert all pressure_value to bar using per-unit conversion rules."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    for col in ["pressure_converted_value", "pressure_converted_unit"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["pressure_converted_value"] = df["pressure_converted_value"].astype("string")
    df["pressure_converted_unit"] = df["pressure_converted_unit"].astype("string")

    # Collect unique units
    all_units = df["pressure_unit"].dropna().unique().tolist()
    has_nan = df["pressure_unit"].isna().any()
    unique_units = sorted(set(str(u).strip() for u in all_units))
    if has_nan:
        unique_units.append("none")
    logger.info(f"[Press] Found {len(unique_units)} unique pressure units")

    # Load / build rules
    rules_path = Path(csv_path).parent / "pressure_conversion_rules.csv"
    if rules_path.exists():
        rules_df = pd.read_csv(rules_path, encoding="utf-8-sig")
        for col in RULES_COLS:
            if col not in rules_df.columns:
                rules_df[col] = None
        done = set(rules_df["original_unit"].dropna().tolist())
        logger.info(f"[Press] Loaded {len(done)} existing rules from checkpoint")
    else:
        rules_df = pd.DataFrame(columns=RULES_COLS)
        done = set()

    pending = [u for u in unique_units if u not in done]
    logger.info(f"[Press] {len(done)} already classified, {len(pending)} pending")

    if pending:
        new_rows = []
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            logger.info(f"[Press] Classifying batch {i//BATCH_SIZE + 1}/{(len(pending)-1)//BATCH_SIZE + 1}: {len(batch)} units")
            try:
                lines = _call_llm_press(batch)
            except Exception as e:
                logger.error(f"[Press] LLM call failed: {e}, retrying in 5s...")
                time.sleep(5)
                try:
                    lines = _call_llm_press(batch)
                except Exception as e2:
                    logger.error(f"[Press] Retry also failed: {e2}, skipping batch")
                    continue

            for line in lines:
                unit, operation, factor = _parse_rule(line)
                if unit and operation:
                    new_rows.append({"original_unit": unit, "operation": operation, "factor": factor})
                    logger.info(f"  [Press] {unit[:60]} → {operation} | {factor}")
                else:
                    logger.warning(f"  [Press] Could not parse: {line[:100]}")

            if new_rows:
                batch_df = pd.DataFrame(new_rows)
                rules_df = pd.concat([rules_df, batch_df], ignore_index=True)
                rules_df = rules_df[RULES_COLS]
                rules_df.to_csv(rules_path, index=False, encoding="utf-8-sig")
                new_rows = []
            time.sleep(0.5)

    # Apply rules
    rules_df = pd.read_csv(rules_path, encoding="utf-8-sig")
    rule_map = {}
    for _, row in rules_df.iterrows():
        rule_map[str(row["original_unit"]).strip()] = (str(row["operation"]).strip(), str(row["factor"]).strip())

    converted_count = 0; identity_count = 0; kept_count = 0
    for idx in df.index:
        raw_unit = df.at[idx, "pressure_unit"]
        raw_value = df.at[idx, "pressure_value"]
        if pd.isna(raw_unit):
            unit_key = "none"
        else:
            unit_key = str(raw_unit).strip()

        if unit_key not in rule_map:
            match = next((k for k in rule_map if k.lower() == unit_key.lower()), None)
            if match:
                unit_key = match
            else:
                logger.warning(f"[Press] No rule for unit '{unit_key}', keeping as-is")
                df.at[idx, "pressure_converted_value"] = str(raw_value) if pd.notna(raw_value) else ""
                df.at[idx, "pressure_converted_unit"] = unit_key
                kept_count += 1
                continue

        operation, factor = rule_map[unit_key]
        if pd.isna(raw_value):
            df.at[idx, "pressure_converted_value"] = ""
            df.at[idx, "pressure_converted_unit"] = "bar" if operation in ("multiply", "identity") else unit_key
            continue

        converted_val, converted_unit = _apply_rule(raw_value, operation, factor, unit_key, target_unit="bar")
        if operation == "multiply":
            converted_count += 1
        elif operation == "identity":
            identity_count += 1
        else:
            kept_count += 1
        df.at[idx, "pressure_converted_value"] = converted_val
        df.at[idx, "pressure_converted_unit"] = converted_unit

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"[Press] [OK] Pressure conversion complete: {converted_count} converted, {identity_count} already bar, {kept_count} kept as-is")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Convert diffusion coefficient units to m²/s (per-unit rules, NOT per-row LLM)")
    parser.add_argument("--input", required=True, help="Path to consolidated CSV file")
    args = parser.parse_args()
    convert_unit(args.input)
