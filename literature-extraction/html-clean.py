#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
html2txt_with_tables.py
Springer/Nature HTML -> readable plain text (body text + tables inserted at reference positions)
"""
import re
import time
import traceback
from pathlib import Path
from lxml import html, etree
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
    base_url="https://api.deepseek.com"
)

MAX_RETRY = 3
SLEEP_RETRY = 5

# ---------- Utilities ----------
def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def table_to_md(table_html: str, counter: int) -> str:
    prompt = f"""Convert the HTML table to compact, aligned Markdown.
Include caption/footnotes/units.  If empty reply "Empty Content".
End with "<END>".

{table_html}
"""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            out = resp.choices[0].message.content.strip()
            if out.endswith("<END>"):
                out = out[:-5].strip()
            if "Empty Content" in out:
                return ""
            return f"\n[Table {counter}]\n{out}\n"
        except Exception as e:
            print(f"[WARN] table llm fail {attempt}/{MAX_RETRY}: {e}")
            time.sleep(SLEEP_RETRY)
    return f"\n[Table {counter}]\n"

# ---------- Table Extraction ----------
def extract_table_md(table_path: Path) -> str:
    html_text = table_path.read_text(encoding='utf8')
    match = re.search(r'<table[^>]*>.*?</table>', html_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    table_html = match.group(0)
    # Extract number from filename
    m = re.search(r'table_(\d+)\.html', table_path.name, re.I)
    idx = int(m.group(1)) if m else 1
    return table_to_md(table_html, idx)

def extract_all_tables(folder: Path) -> dict[int, str]:
    tables = {}
    for table_file in folder.glob("table_*.html"):
        m = re.match(r'table_(\d+)\.html', table_file.name, re.I)
        if not m:
            continue
        idx = int(m.group(1))
        tables[idx] = extract_table_md(table_file)
    return tables

# ---------- Insert Tables ----------
def insert_tables_into_text(lines: list[str], tables: dict[int, str]) -> list[str]:
    inserted = set()
    new_lines = []
    for line in lines:
        new_lines.append(line)
        for m in re.finditer(r'\bTable\s+(\d+)', line, re.I):
            idx = int(m.group(1))
            if idx in tables and idx not in inserted:
                new_lines.append(tables[idx])
                inserted.add(idx)
    # Append unreferenced tables
    for idx in sorted(tables):
        if idx not in inserted:
            new_lines.append(tables[idx])
    return new_lines

# ---------- Process Single Folder ----------
def process_folder(folder: Path):
    print(f"\n[INFO] Processing folder: {folder}")
    main_html = folder / "main.html"
    out_txt = folder / f"{folder.name}.txt"

    if not main_html.exists():
        print(f"[SKIP] No main.html in {folder}")
        return

    try:
        root = html.parse(main_html).getroot()
        sections = root.xpath("//section[starts-with(@id,'Sec') or @id='Abs1-section']")
        if not sections:
            sections = root.xpath("//div[@data-article-body='true']//section")
        if not sections:
            print(f"[WARN] No content in {main_html}")
            return

        lines = []
        for sec in sections:
            for para in sec.xpath(".//p"):
                txt = clean_space("".join(para.itertext()))
                if txt:
                    lines.append(txt)

        tables = extract_all_tables(folder)
        final_lines = insert_tables_into_text(lines, tables)
        final = "\n\n".join(final_lines)
        out_txt.write_text(final, encoding='utf8')
        print(f"[OK] {out_txt}")
    except Exception:
        traceback.print_exc()

# ---------- Main Entry Point ----------
def walk_and_convert(input_dir: Path):
    # Check if this is a single document folder (contains main.html)
    if (input_dir / "main.html").exists():
        process_folder(input_dir)
        return

    # Otherwise traverse subdirectories
    for item in input_dir.iterdir():
        if item.is_dir():
            process_folder(item)

# ---------- CLI ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python html-clean.py <parent_folder>")
        sys.exit(1)
    parent = Path(sys.argv[1])
    if not parent.exists():
        print("Folder not found")
        sys.exit(1)
    walk_and_convert(parent)