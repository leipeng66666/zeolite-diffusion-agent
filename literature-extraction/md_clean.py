# -*- coding: utf-8 -*-
"""Clean PaddleOCR-generated MD files before sending to value.py.

Fixes (stage: pre-processing, before value.py):
  1. Convert <html><body><table> to readable [Table] format
  2. Remove image references: ![](images/...)
  3. Collapse excessive blank lines (3+ → 1)
"""

import re, os, glob, argparse


def _html_table_to_text(html_line: str) -> str:
    """Convert a single <html><body><table>... line to readable pipe table."""
    # Strip outer wrapper
    inner = re.sub(r'</?html>|</?body>', '', html_line, flags=re.IGNORECASE)
    # Split by table rows
    rows = re.findall(r'<tr>(.*?)</tr>', inner, re.IGNORECASE)
    if not rows:
        return html_line

    lines = []
    for row in rows:
        # Match <td> with optional attributes (colspan, rowspan)
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.IGNORECASE)
        lines.append(' | '.join(cells))

    if not lines:
        return html_line

    return '[Table]\n' + '\n'.join(lines)


def clean_md_text(text: str) -> str:
    """Clean a single MD document's text content."""
    # 1. Convert <html><body><table> blocks to readable format
    text = re.sub(
        r'<html><body><table>.*?</table></body></html>',
        lambda m: _html_table_to_text(m.group(0)),
        text,
        flags=re.IGNORECASE
    )

    # 2. Remove image references: ![](images/xxx.jpg) or ![alt](images/xxx.jpg)
    text = re.sub(r'!\[.*?\]\(images/[^)]+\)', '', text)

    # 3. Collapse 3+ consecutive blank lines to 1 blank line
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4. Strip trailing whitespace per line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))

    # 5. Remove empty lines at start/end
    text = text.strip() + '\n'

    return text


def clean_md_file(input_path: str, output_path: str = None) -> str:
    """Clean a single MD file. If output_path is None, overwrite input."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    cleaned = clean_md_text(content)

    target = output_path or input_path
    with open(target, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    return target


def batch_clean(input_dir: str, output_dir: str = None):
    """Clean all MD files in a directory."""
    out = output_dir or input_dir
    os.makedirs(out, exist_ok=True)

    md_files = glob.glob(os.path.join(input_dir, "*.md"))
    for f in md_files:
        base = os.path.basename(f)
        out_path = os.path.join(out, base)
        clean_md_file(f, out_path)

    print(f"Cleaned {len(md_files)} MD files: {input_dir} -> {out}")
    return len(md_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean PaddleOCR MD files")
    parser.add_argument("--input", required=True, help="Input directory with .md files")
    parser.add_argument("--output", help="Output directory (default: overwrite input)")
    args = parser.parse_args()
    batch_clean(args.input, args.output)
