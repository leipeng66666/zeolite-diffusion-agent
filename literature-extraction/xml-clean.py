'''Convert the XML files into readable text documents,
now capable of handling more complex headers and improving the extraction of footnotes.
Change Log:
Date        | Author              | Description
------------|---------------------|---------------------------------------------
2025/05/15  | lp                | Basic functionality implemented'''
import re
from lxml import etree
from html import unescape
import os
import glob
import traceback
import sys
from openai import OpenAI
from time import sleep
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
    base_url="https://api.deepseek.com"
)

def mathml_to_text(math_element):
    """Convert MathML to LaTeX-like text, preserving inter-element text and tails."""

    def process_node(node):
        tag = etree.QName(node).localname
        if tag in ('mi', 'mn', 'mo', 'rm'):
            return node.text or ''
        elif tag == 'msub':
            base = process_node(node[0]) if len(node) > 0 else ''
            sub = process_node(node[1]) if len(node) > 1 else ''
            return f"{base}_{{{sub}}}"
        elif tag == 'msup':
            base = process_node(node[0]) if len(node) > 0 else ''
            sup = process_node(node[1]) if len(node) > 1 else ''
            return f"{base}^{{{sup}}}"
        elif tag in ('inf', 'sub'):
            inner = _flatten_math(node)
            return f"_{{{inner}}}"
        elif tag in ('sup',):
            inner = _flatten_math(node)
            return f"^{{{inner}}}"
        elif tag == 'mfrac':
            num = process_node(node[0]) if len(node) > 0 else ''
            den = process_node(node[1]) if len(node) > 1 else ''
            return f"({num}/{den})"
        elif tag == 'mrow':
            return _flatten_math(node)
        elif tag == 'mfenced':
            return '(' + ', '.join(process_node(child) for child in node) + ')'
        elif tag in ('hsp',):
            return ' '
        return ''

    def _flatten_math(elem):
        """Recursively flatten a math element, preserving text and tails."""
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            parts.append(process_node(child))
            if child.tail:
                parts.append(child.tail)
        return ''.join(parts)

    return _flatten_math(math_element)


def process_element(elem, ns):

    text = []
    if elem.text and elem.text.strip():
        text.append(elem.text.strip())

    for child in elem:
        tag = etree.QName(child).localname
        if tag == 'inf':
            text.append(f"_{{{process_element(child, ns)}}}")
        elif tag == 'sup':
            text.append(f"^{{{process_element(child, ns)}}}")
        elif tag == 'cross-ref':
            ref_id = child.get('refid', '')
            ref_text = re.sub(r'\D', '', ref_id)
            text.append(f"[Ref:{ref_text}]")
        else:
            text.append(process_element(child, ns))

        if child.tail and child.tail.strip():
            text.append(child.tail.strip())

    return ''.join(text).replace('\n', ' ').strip()


def convert_paragraph(para_element, ns):

    text = []
    if para_element.text and para_element.text.strip():
        text.append(para_element.text.strip())

    for child in para_element:
        if etree.QName(child).localname == 'math':
            math_text = mathml_to_text(child)
            text.append(f" {math_text} ")
        else:
            processed = process_element(child, ns)
            if processed:
                text.append(processed)

        if child.tail and child.tail.strip():
            text.append(child.tail.strip())

    final_text = unescape(' '.join(text))
    return re.sub(r'\s+', ' ', final_text).strip()


def process_table(table_element, ns):
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Get the full table XML content
            table_xml = etree.tostring(table_element, encoding='unicode', pretty_print=True)

            # Build LLM request
            messages = [
                {
                    "role": "system",
                    "content": """Your goal is to convert the provided XML/HTML table into a clear and concise Markdown format,
ensuring that essential information, such as title label, captions and footnotes, is not omitted. without any residual XML tags such as<sup>
If there is no content except for the Footnote, return 'Empty Content'.
You must conclude with '<END>'."""
                },
                {
                    "role": "user",
                    "content": f"Please adjust the table format, remove all unnecessary spaces, and ensure that the data is aligned and displayed compactly..Unit conversion to readable format.Convert this table:\n{table_xml}"
                }
            ]

            # Call LLM
            response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=messages,
                temperature=0
            )

            # Process response content
            converted = response.choices[0].message.content.strip()

            # Clean end marker
            if converted.endswith('<END>'):
                converted = converted[:-5].strip()

            # Handle empty content case
            if 'Empty Content' in converted:
                return ""

            # Add table prefix identifier
            table_id = table_element.get('id') or ''
            clean_id = re.sub(r'\D', '', table_id).zfill(3)
            return f"\n[Table {clean_id}]\n{converted}\n"

        except Exception as e:
            print(f"Table conversion failed: {str(e)}, retry {retry_count+1}/{max_retries}")
            traceback.print_exc()
            retry_count += 1
            sleep(5)

    print(f"Table conversion failed, max retries reached: {max_retries}")
    return ""


def process_xml(xml_path, output_path):

    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        ns = {'ce': root.nsmap.get('ce', '')}

        # Pre-load tables (using sequential numbering)
        tables = {}
        for idx, table in enumerate(root.xpath('//ce:table', namespaces=ns), 1):
            table_id = f"Table_{idx}"  # Use sequential numbering
            print(f"Table preprocessing: seq_number={table_id}")
            tables[table_id] = table

        all_content = []
        processed_tables = set()
        table_ref_positions = {}

        # Extract figure captions (often contain data values like D, T, P)
        # Assign them to positions near where the figure is referenced
        figure_captions = {}
        for fig in root.xpath('//ce:figure', namespaces=ns):
            fig_id = fig.get('id', '')
            cap_nodes = fig.xpath('.//ce:caption//ce:simple-para', namespaces=ns)
            caption_parts = []
            for cap in cap_nodes:
                cap_text = convert_paragraph(cap, ns)
                if cap_text:
                    caption_parts.append(cap_text)
            if caption_parts:
                label = fig.find('.//ce:label', ns)
                label_text = process_element(label, ns) if label is not None else ''
                figure_captions[fig_id] = f"[Figure] {label_text} {' '.join(caption_parts)}"

        # First pass: collect all table reference positions and figure reference positions
        figure_ref_positions = {}
        paras = root.xpath('//ce:para', namespaces=ns)
        for para_idx, para in enumerate(paras, 1):
            para_text = convert_paragraph(para, ns)

            # Detect table references (supports multiple reference formats)
            matches = re.finditer(
                r'\b(?<!ESI\s)(?<!S\s)(?:Table|Tables)\s+(\d+)(?:\s*,\s*(\d+))*(?:\s+and\s+(\d+))?(?=\b|,)',
                para_text,
                flags=re.IGNORECASE
            )

            for match in matches:
                refs = []
                if match.group(1):
                    refs.append(f"Table_{match.group(1)}")
                if match.group(2):
                    refs.append(f"Table_{match.group(2)}")
                if match.group(3):
                    refs.append(f"Table_{match.group(3)}")

                for ref in refs:
                    if ref in tables:
                        if ref not in table_ref_positions:
                            table_ref_positions[ref] = []
                        table_ref_positions[ref].append(para_idx)
                        print(f"Found reference to {ref} in paragraph {para_idx}")

            # Detect figure references (e.g., "Fig. 5", "Figure 5", "Figs. 5 and 6")
            fig_matches = re.finditer(
                r'\b(?:Fig|Figs|Figure|Figures)\.?\s+(\d+)(?:\s*,\s*(\d+))*(?:\s+and\s+(\d+))?',
                para_text, flags=re.IGNORECASE
            )
            for fm in fig_matches:
                for g in fm.groups():
                    if g:
                        fid = f"FIG{g}"
                        if fid in figure_captions:
                            if fid not in figure_ref_positions:
                                figure_ref_positions[fid] = []
                            figure_ref_positions[fid].append(para_idx)
                            print(f"Found reference to figure {fid} in paragraph {para_idx}")

        # Second pass: insert tables and figures at last reference position
        processed_figures = set()
        for para_idx, para in enumerate(paras, 1):
            para_text = convert_paragraph(para, ns)
            all_content.append(para_text)

            # Check if current paragraph is the last reference position for any table
            for table_id, ref_positions in table_ref_positions.items():
                if para_idx in ref_positions and para_idx == max(ref_positions):
                    if table_id in tables and table_id not in processed_tables:
                        try:
                            table_text = process_table(tables[table_id], ns)
                            if table_text:
                                table_text = table_text.replace("[Table ", f"[{table_id} ")
                                all_content.append(table_text)
                                processed_tables.add(table_id)
                                print(f"Successfully inserted {table_id} after paragraph {para_idx}")
                            else:
                                print(f"Table {table_id} has empty content")
                        except Exception as e:
                            print(f"Table processing error: {table_id} - {str(e)}")
                            traceback.print_exc()

            # Insert figure captions at last reference position
            for fid, ref_positions in figure_ref_positions.items():
                if para_idx in ref_positions and para_idx == max(ref_positions):
                    if fid in figure_captions and fid not in processed_figures:
                        all_content.append(figure_captions[fid])
                        processed_figures.add(fid)
                        print(f"Successfully inserted figure {fid} after paragraph {para_idx}")

        # Process unreferenced tables (in document order)
        for table_id in sorted(tables.keys()):
            if table_id not in processed_tables:
                try:
                    table_text = process_table(tables[table_id], ns)
                    if table_text:
                        table_text = table_text.replace("[Table ", f"[{table_id} ")
                        all_content.append(table_text)
                        processed_tables.add(table_id)
                        print(f"Appended unreferenced table {table_id}")
                except Exception as e:
                    print(f"Unreferenced table processing error: {table_id} - {str(e)}")

        # Append unreferenced figure captions at end
        for fid, cap_text in figure_captions.items():
            if fid not in processed_figures:
                all_content.append(cap_text)
                processed_figures.add(fid)
                print(f"Appended unreferenced figure {fid}")

        # Write to file
        final_output = '\n\n'.join(all_content)
        print(f"Final content preview (first 500 chars):\n{final_output[:500]}...")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_output)

        print(f"Successfully written: {output_path}")
        return True

    except Exception as e:
        print(f"Error processing file: {xml_path}\n{str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input XML directory")
    parser.add_argument("--output", required=True, help="Output TXT directory")
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    xml_files = glob.glob(os.path.join(input_folder, "*.xml"))
    print(f"Found {len(xml_files)} XML files")

    success_count = 0
    for idx, xml_file in enumerate(xml_files, 1):
        print(f"\nProcessing file ({idx}/{len(xml_files)}) {os.path.basename(xml_file)}")
        output_name = os.path.splitext(os.path.basename(xml_file))[0] + ".txt"
        output_path = os.path.join(output_folder, output_name)

        if process_xml(xml_file, output_path):
            success_count += 1

    print(f"\nProcessing complete! Successfully converted {success_count}/{len(xml_files)} files")
