"""
agent.py - Zeolite Diffusion Data Processing Agent System
Supports XML files, HTML folders, and Markdown file processing
XML/HTML -> TXT(clean) -> extraction stage
Markdown -> directly enters extraction stage (no cleaning needed)
"""
import os
import sys
import subprocess
import time
import argparse
import shutil
import glob
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from pathlib import Path
from md_clean import clean_md_text

# Force UTF-8 on Windows terminal to avoid GBK decode errors from emoji output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# -- Import stage processing functions ----------------------------------------
try:
    import importlib.util, types
    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _BASE = Path(__file__).parent
    _value_mod   = _load("value",   _BASE / "value.py")
    _multi_mod   = _load("multi",   _BASE / "Multidimensional-Data.py")
    _unit_mod    = _load("unit",    _BASE / "unit_converter.py")
    _uniq_mod    = _load("uniq",    _BASE / "unique_extractor.py")
    _norm_mod    = _load("norm",    _BASE / "normalizer.py")
    _map_mod     = _load("mapper",  _BASE / "mapper.py")
    _method_mod  = _load("method",  _BASE / "method_classifier.py")
    _clean_mod   = _load("clean",   _BASE / "cleaned_output.py")
    _xml_mod     = _load("xmlclean", _BASE / "xml-clean.py")
    _html_mod    = _load("htmlclean", _BASE / "html-clean.py")

    process_txt_strict                     = _value_mod.process_txt_strict
    normalize_diffusion_value              = _value_mod.normalize_diffusion_value
    process_txt_file_with_diffusion_values = _multi_mod.process_txt_file_with_diffusion_values
    convert_unit   = _unit_mod.convert_unit
    convert_temperature_to_kelvin = _unit_mod.convert_temperature_to_kelvin
    convert_concentration_to_mol_per_l = _unit_mod.convert_concentration_to_mol_per_l
    convert_pressure_to_bar = _unit_mod.convert_pressure_to_bar
    extract_unique = _uniq_mod.extract_unique
    normalize_all  = _norm_mod.normalize_all
    map_back       = _map_mod.map_back
    classify_methods = _method_mod.classify_methods
    generate_cleaned_csv = _clean_mod.generate_cleaned_csv
    _process_xml          = _xml_mod.process_xml
    _process_html_folder  = _html_mod.process_folder
    _MODULES_OK = True
except Exception as _e:
    _MODULES_OK = False
    _MODULE_ERR = str(_e)

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('ZeoliteAgent')


def _move_to_no_data(file_path, document_dir):
    """Move TXT/CSV/JSON/XML files for a paper with no valid data to no_data/ folder at project root."""
    base = os.path.splitext(os.path.basename(file_path))[0]
    project_dir = os.path.dirname(document_dir)
    no_data_dir = os.path.join(project_dir, 'no_data')
    os.makedirs(no_data_dir, exist_ok=True)

    for ext in ('.txt', '.csv', '.json', '.xml'):
        src = os.path.join(document_dir, base + ext)
        if os.path.exists(src):
            dst = os.path.join(no_data_dir, base + ext)
            os.replace(src, dst)
            logger.info(f"  Moved {base}{ext} -> no_data/")
if not _MODULES_OK:
    logger.warning(f"Module import failed, will fallback to subprocess mode: {_MODULE_ERR}")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


class ProcessingAgent:
    def __init__(self, workspace):
        # Get and normalize workspace path
        self.workspace = os.path.abspath(workspace)
        logger.info(f"Initializing workspace: {self.workspace}")

        # Ensure workspace exists
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
            logger.info(f"Created new workspace: {self.workspace}")

        # Define key paths
        self.document_source = os.path.join(self.workspace, "Document")
        self.markdown_source = os.path.join(self.workspace, "Markdown")  # Markdown folder, md goes directly to extraction
        self.xml_clean_script = os.path.join(self.workspace, "xml-clean.py")
        self.multidimensional_data_script = os.path.join(self.workspace, "Multidimensional-Data.py")
        self.value_script = os.path.join(self.workspace, "value.py")
        self.html_script = os.path.join(self.workspace, "html-clean.py")
        self.final_output = os.path.join(self.workspace, "consolidated_results3.csv")
        self.log_dir = os.path.join(self.workspace, "logs")

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Validate scripts exist
        self.validate_scripts()

        # Create necessary directories and symbolic links
        self.setup_workspace()

        # Log environment information
        self.log_environment()

    def log_environment(self):
        """Log current environment information"""
        logger.info("="*50)
        logger.info("Environment Info:")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.executable}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"System path: {os.environ.get('PATH', '')}")
        logger.info("="*50)

    def validate_scripts(self):
        """Verify required scripts exist in the workspace"""
        required_scripts = {
            "xml-clean.py": self.xml_clean_script,
            "Multidimensional-Data.py": self.multidimensional_data_script,
            "value.py": self.value_script,
            "html-clean.py": self.html_script
        }

        missing = []
        for name, path in required_scripts.items():
            if not os.path.exists(path):
                missing.append(name)

        if missing:
            logger.error(f"Missing required scripts: {', '.join(missing)}")
            logger.error(f"Please copy these files to the workspace: {self.workspace}")
            sys.exit(1)

    def setup_workspace(self):
        """Create workspace structure"""
        os.makedirs(self.document_source, exist_ok=True)
        os.makedirs(self.markdown_source, exist_ok=True)  # Ensure Markdown directory exists
        link_path = os.path.join(self.workspace, "Document")

        if not os.path.exists(link_path):
            try:
                os.symlink(self.document_source, link_path)
                logger.info(f"Created symbolic link: {link_path} -> {self.document_source}")
            except OSError as e:
                logger.warning(f"Cannot create symbolic link: {e}")
                shutil.copytree(self.document_source, link_path)
                logger.info(f"Created directory copy: {link_path}")

    def should_skip_doc1(self):
        """
        Check whether to skip step 1 (XML and HTML to TXT conversion)
        - If Document/ has no XML files and no HTML folders -> skip step 1 directly
        - If XML/HTML files exist but all have corresponding TXT -> skip step 1
        - Otherwise need to execute step 1
        Note: Markdown files do not participate in this check; md files do not need cleaning and go directly to extraction
        """
        xml_files = [f for f in os.listdir(self.document_source) if f.endswith('.xml')]
        html_folders = [d for d in os.listdir(self.document_source)
                        if os.path.isdir(os.path.join(self.document_source, d)) and
                        os.path.exists(os.path.join(self.document_source, d, "main.html"))]
        # No XML or HTML folders, no need for cleaning step
        if not xml_files and not html_folders:
            logger.info("No XML/HTML files in Document/, skipping cleaning step")
            return True

        missing_txt = []
        for x in xml_files:
            if not os.path.exists(os.path.join(self.document_source, x[:-4] + '.txt')):
                missing_txt.append(x)
        for d in html_folders:
            if not os.path.exists(os.path.join(self.document_source, d + '.txt')):
                missing_txt.append("HTML-" + d)

        if missing_txt:
            return False
        logger.info("All source files already have corresponding TXT files, skipping step 1")
        return True

    def run_markdown_to_doc(self):
        """
        Copy .md files from Markdown/ directory directly to Document/ directory.
        md files do not need cleaning; they participate directly in the extraction stage.
        If an .md file with the same name already exists in Document/, it will be overwritten.
        """
        md_files = glob.glob(os.path.join(self.markdown_source, "*.md"))
        if not md_files:
            logger.info("No .md files in Markdown/ directory, skipping")
            return True
        logger.info(f"Found {len(md_files)} .md files, copying directly to Document/ for extraction")
        ok = True
        for src in md_files:
            fname = os.path.basename(src)
            dst = os.path.join(self.document_source, fname)
            try:
                shutil.copy2(src, dst)
                logger.info(f"Copied: {fname} -> Document/")
            except Exception as e:
                logger.error(f"[FAIL] Copy {fname} failed: {e}")
                ok = False
        return ok

    def run_script(self, script_name, args, custom_cwd=None):
        """Generic script execution method"""
        cmd = [sys.executable, script_name] + args
        logger.info(f"Executing command: {' '.join(cmd)}")
        script_base = os.path.basename(script_name)
        log_file = os.path.join(self.log_dir, f"{script_base}_{int(time.time())}.log")
        cwd = custom_cwd if custom_cwd else self.workspace
        with open(log_file, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8"
            )
            stdout = []
            for line in process.stdout:
                log.write(line)
                stdout.append(line)
                logger.info(f"[{script_base}] {line.rstrip()}")
            ret = process.wait()
            if ret == 0:
                logger.info(f"[OK] Script succeeded: {script_name}")
                return True
            else:
                logger.error(f"[FAIL] Script failed: {script_name} exit code {ret}")
                return False

    def run_html_processing(self, folder_path):
        """Process HTML folder"""
        main_html = os.path.join(folder_path, "main.html")
        if not os.path.exists(main_html):
            logger.error(f"[FAIL] HTML folder missing main.html: {folder_path}")
            return False
        folder_name = os.path.basename(folder_path)
        logger.info(f"Processing HTML folder: {folder_path}")
        ok = self.run_script(self.html_script, [folder_path])
        if not ok:
            return False
        expected_txt = folder_name + ".txt"
        src = os.path.join(folder_path, expected_txt)
        if not os.path.exists(src):
            src = os.path.join(folder_path, "cleaned.txt")
        if not os.path.exists(src):
            logger.error(f"[FAIL] HTML processing did not generate TXT file: {folder_path}")
            return False
        dst = os.path.join(self.document_source, expected_txt)
        shutil.move(src, dst)
        logger.info(f"Moved TXT to Document source directory: {dst}")
        return True

    def run_doc1(self):
        """Step 1: XML & HTML -> TXT (Markdown files are copied directly to Document/ beforehand, no cleaning needed)"""
        logger.info("\n" + "="*50 + "\nStep 1: Convert XML and HTML to TXT\n" + "="*50)
        xml_ok = self.run_script(self.xml_clean_script,
                ["--input", self.document_source, "--output", self.document_source])
        html_ok = True
        html_dirs = [os.path.join(self.document_source, d) for d in os.listdir(self.document_source)
                     if os.path.isdir(os.path.join(self.document_source, d)) and
                     os.path.exists(os.path.join(self.document_source, d, "main.html"))]
        for d in html_dirs:
            if not self.run_html_processing(d):
                html_ok = False
        return xml_ok and html_ok

    # -- Per-file processing core ---------------------------------------------------
    def _get_done_filenames(self):
        """Read completed filename set from consolidated CSV (checkpoint resume)"""
        done = set()
        if os.path.exists(self.final_output):
            try:
                df = pd.read_csv(self.final_output)
                if 'filename' in df.columns:
                    done = set(df['filename'].astype(str).tolist())
            except Exception:
                pass
        return done

    def _clean_md(self, md_path: str) -> str:
        """Clean a PaddleOCR MD file before extraction (remove images, collapse blank lines)."""
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        cleaned = clean_md_text(content)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        return md_path

    def _clean_one(self, src_path: str):
        """
        Clean a single source file (XML or HTML folder), return the generated TXT path.
        If corresponding TXT already exists, return it directly (checkpoint resume).
        Returns None on failure.
        """
        base    = os.path.splitext(os.path.basename(src_path))[0]
        out_txt = os.path.join(self.document_source, base + ".txt")

        if os.path.exists(out_txt):
            logger.info(f"[SKIP] [{base}] TXT already exists, skipping cleaning")
            return out_txt

        if src_path.endswith(".xml"):
            logger.info(f"[->] [{base}] XML -> TXT ...")
            ok = _process_xml(src_path, out_txt)
            if ok and os.path.exists(out_txt):
                logger.info(f"[OK] [{base}] XML cleaning completed")
                return out_txt
            logger.error(f"[FAIL] [{base}] XML cleaning failed")
            return None

        if os.path.isdir(src_path) and os.path.exists(os.path.join(src_path, "main.html")):
            logger.info(f"[->] [{base}] HTML -> TXT ...")
            try:
                _process_html_folder(Path(src_path))
                inner = os.path.join(src_path, base + ".txt")
                if os.path.exists(inner):
                    shutil.move(inner, out_txt)
                    logger.info(f"[OK] [{base}] HTML cleaning completed")
                    return out_txt
                logger.error(f"[FAIL] [{base}] HTML did not produce TXT")
            except Exception as exc:
                logger.error(f"[FAIL] [{base}] HTML cleaning exception: {exc}")
            return None

        logger.warning(f"[WARN] Unknown source file type, skipping: {src_path}")
        return None

    def process_single_file(self, src_file: str) -> bool:
        """
        Run all steps for a single source file (atomic per-file pipeline):
          Step 0: XML/HTML -> TXT (MD files skip this step directly)
          Step A: Extract diffusion coefficient values -> per-file CSV
          Step B: Extract detailed multi-dimensional data -> append to consolidated CSV
        """
        import re as _re
        ext  = os.path.splitext(src_file)[1].lower()
        base = os.path.splitext(os.path.basename(src_file))[0]
        logger.info(f"\n" + "="*50 + f"\n[Start Processing] {base}\n" + "="*50)

        # -- Step 0: Clean to TXT (XML/HTML only) ------------------------------------
        if ext == ".xml" or (os.path.isdir(src_file) and
                              os.path.exists(os.path.join(src_file, "main.html"))):
            text_file = self._clean_one(src_file)
            if not text_file:
                return False
        elif ext == ".md":
            # Clean MD before extraction
            text_file = self._clean_md(src_file)
        else:
            text_file = src_file  # .txt used directly

        per_csv   = os.path.join(self.document_source, base + ".csv")
        logger.info(f"\n--- Extraction Stage: {base} ---")

        # -- Step A: Extract diffusion coefficients ----------------------------------
        if not os.path.exists(per_csv):
            try:
                results = process_txt_strict(text_file)
            except Exception as exc:
                logger.error(f"[FAIL] value extraction exception [{base}]: {exc}")
                return False

            if not results:
                logger.warning(f"[WARN] No extraction results: {base}")
                # Write empty CSV placeholder to avoid reprocessing
                pd.DataFrame(columns=['diffusion_value']).to_csv(per_csv, index=False, encoding="utf-8-sig")
                _move_to_no_data(text_file, self.document_source)
                return False

            df_v = pd.DataFrame(results)
            if not df_v.empty:
                def _clean(v):
                    normed = normalize_diffusion_value(v)
                    return normed if normed is not None else v
                df_v['diffusion_value'] = df_v['diffusion_value'].apply(_clean)
                # Filter: keep pure numbers and scientific notation (e/E), exclude unit letters and none
                def _is_valid_val(v):
                    v = str(v).strip()
                    if v.lower() == 'none' or v == '':
                        return False
                    import re as _re2
                    return bool(_re2.fullmatch(r'[+-]?[0-9]+(?:[.,][0-9]+)?(?:[eE][+-]?[0-9]+)?', v))
                df_v = df_v[df_v['diffusion_value'].apply(_is_valid_val)]

            if df_v.empty:
                logger.warning(f"[WARN] No valid diffusion coefficient data: {base}")
                pd.DataFrame(columns=['diffusion_value']).to_csv(per_csv, index=False, encoding="utf-8-sig")
                _move_to_no_data(text_file, self.document_source)
                return False

            df_v.to_csv(per_csv, index=False, encoding="utf-8-sig")
            logger.info(f"[OK] [{base}] Diffusion coefficient extraction complete, {len(df_v)} entries")
        else:
            logger.info(f"[SKIP] [{base}] per-file CSV already exists, skipping diffusion coefficient extraction")

        # -- Step B: Extract detailed data -------------------------------------------
        try:
            df_per = pd.read_csv(per_csv)
        except Exception as exc:
            logger.error(f"[FAIL] Failed to read per-file CSV [{base}]: {exc}")
            return False

        if 'diffusion_value' not in df_per.columns or df_per.empty:
            logger.warning(f"[WARN] [{base}] per-file CSV has no valid diffusion coefficients, skipping detailed extraction")
            _move_to_no_data(text_file, self.document_source)
            return False

        diffusion_values = df_per['diffusion_value'].tolist()
        try:
            ok = process_txt_file_with_diffusion_values(text_file, diffusion_values, self.final_output)
        except Exception as exc:
            logger.error(f"[FAIL] Multidimensional extraction exception [{base}]: {exc}")
            return False

        if ok:
            logger.info(f"[OK] [{base}] Detailed data written to consolidated CSV")
        else:
            logger.warning(f"[WARN] [{base}] No detailed data extracted (LLM did not return valid records, not written to consolidated CSV)")
        return ok

    def run_doc3(self):
        """Step 2: Extract diffusion coefficient values (process .txt and .md files)"""
        logger.info("\n" + "="*50 + "\nStep 2: Extract diffusion coefficient values\n" + "="*50)
        txt_files = glob.glob(os.path.join(self.document_source, "*.txt"))
        md_files = glob.glob(os.path.join(self.document_source, "*.md"))
        if not txt_files and not md_files:
            logger.info("No TXT/MD files, skipping")
            return True
        return self.run_script(self.value_script,
                               ["--input", self.document_source, "--output", self.document_source])

    def run_doc2(self):
        """Step 3: Extract detailed data"""
        logger.info("\n" + "="*50 + "\nStep 3: Extract detailed data\n" + "="*50)
        csv_files = glob.glob(os.path.join(self.document_source, "*.csv"))
        if not csv_files:
            logger.error("[FAIL] No CSV files in input directory")
            return False
        ok = self.run_script(self.multidimensional_data_script,
                ["--input", self.document_source, "--output", self.final_output])
        if ok and os.path.exists(self.final_output):
            df = pd.read_csv(self.final_output)
            logger.info(f"[OK] Final output has {len(df)} records")
            df.to_csv(self.final_output, index=False, encoding="utf-8-sig")
            return True
        else:
            logger.error("[FAIL] Final output file not generated")
            return False

    def full_workflow(self):
        """Full workflow: per-file atomic processing (including XML/HTML cleaning)"""
        logger.info("\n" + "="*60 + f"\nStart processing workspace: {self.workspace}\n" + "="*60)

        if not _MODULES_OK:
            logger.error(f"Core module loading failed, cannot run: {_MODULE_ERR}")
            return False

        # 1. Copy Markdown/ to Document/
        self.run_markdown_to_doc()

        # 2. Collect all source files (XML, HTML folders, TXT, MD), deduplicate and enqueue
        xml_files    = glob.glob(os.path.join(self.document_source, "*.xml"))
        html_folders = [os.path.join(self.document_source, d)
                        for d in os.listdir(self.document_source)
                        if os.path.isdir(os.path.join(self.document_source, d))
                        and os.path.exists(os.path.join(self.document_source, d, "main.html"))]
        txt_files    = glob.glob(os.path.join(self.document_source, "*.txt"))
        md_files     = glob.glob(os.path.join(self.document_source, "*.md"))

        # Deduplicate by priority: XML > HTML folder > TXT > MD (same base_name keeps highest priority)
        seen, ordered = set(), []
        for group in (xml_files, html_folders, txt_files, md_files):
            for f in group:
                base = os.path.splitext(os.path.basename(f))[0]
                if base not in seen:
                    seen.add(base)
                    ordered.append(f)

        if not ordered:
            logger.warning("[WARN] No processable files in Document/")
            return False

        # 3. Checkpoint resume: filter files already written to consolidated CSV
        done    = self._get_done_filenames()
        pending = [f for f in ordered
                   if os.path.splitext(os.path.basename(f))[0] not in done]

        logger.info(f"Total {len(ordered)} files, {len(done)} completed, {len(pending)} pending")

        # 4. Per-file atomic processing (each file goes through cleaning -> extraction -> write)
        success_count = 0
        for i, src in enumerate(pending, 1):
            logger.info(f"\n[{i}/{len(pending)}] {os.path.basename(src)}")
            if self.process_single_file(src):
                success_count += 1
            else:
                logger.warning(f"[WARN] Skipped: {os.path.basename(src)}")

        logger.info(f"\n[OK] Per-file processing complete: {success_count}/{len(pending)} succeeded")

        # 5. Post-processing: unit conversion + normalization + mapping (executed once after full consolidation)
        # Run as long as the consolidated CSV has data — not just when this run added new rows
        has_data = os.path.exists(self.final_output) and os.path.getsize(self.final_output) > 0
        if has_data:
            logger.info("\n" + "="*60 + "\nStarting post-processing: unit conversion -> normalization -> mapping\n" + "="*60)
            try:
                convert_unit(self.final_output)
                convert_temperature_to_kelvin(self.final_output)
                convert_concentration_to_mol_per_l(self.final_output)
                convert_pressure_to_bar(self.final_output)
                extract_unique(self.final_output)
                normalize_all(self.final_output)
                map_back(self.final_output)
                classify_methods(self.final_output)
                generate_cleaned_csv(self.final_output)
                logger.info("[OK] Post-processing complete")
            except Exception as exc:
                logger.warning(f"[WARN] Post-processing exception (does not affect main workflow): {exc}")

        logger.info("\n" + "*"*60 + f"\nAll done! Final result: {self.final_output}\n" + "*"*60)
        return True



def get_workspace():
    while True:
        ws = input("Please enter workspace folder path: ").strip()
        if ws:
            return str(Path(ws).expanduser().resolve())
        print("Input cannot be empty, please try again")

def main():
    parser = argparse.ArgumentParser(description="Zeolite Diffusion Data Processing Agent System")
    parser.add_argument("--workspace", help="Workspace root directory")
    parser.add_argument("--mode", choices=["full", "watch"], default="full")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")

    if not args.workspace:
        args.workspace = get_workspace()

    agent = ProcessingAgent(args.workspace)
    if args.mode == "full":
        agent.full_workflow()
    elif args.mode == "watch":
        # Watch mode omitted, same as original logic
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated")
    except Exception as e:
        logger.exception(f"System error: {str(e)}")
        sys.exit(1)
