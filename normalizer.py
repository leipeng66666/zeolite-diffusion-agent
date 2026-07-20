
import pandas as pd, os, time, logging
from pathlib import Path
from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
    base_url="https://api.deepseek.com"
)

PROMPT_DIR = Path(__file__).with_name("prompts")
logger = logging.getLogger("normalizer")

def _load_prompt(col: str) -> str:
    return (PROMPT_DIR / f"{col}.txt").read_text(encoding="utf-8-sig").strip()

def _llm_normalize_one(text: str, prompt_header: str) -> str:
    """Normalize a single entry, return original string on failure as fallback"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": f"{prompt_header}\n{text}"}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM processing failed: {e}, using fallback")
        return text

def normalize_all(csv_path: str) -> None:
    base = Path(csv_path).parent
    for col in ["zeolite_name", "guest_molecule"]:
        df = pd.read_csv(base / f"{col}_unique.csv", encoding="utf-8-sig")
        prompt = _load_prompt(col)
        std_col = f"std_{col}"

        # Checkpoint resume: skip rows that already have std_ values, only process new rows
        if std_col not in df.columns:
            df[std_col] = None

        need_process = df[std_col].isna() | (df[std_col].astype(str).str.strip() == '')
        todo_idx = df.index[need_process].tolist()
        skip_count = len(df) - len(todo_idx)
        logger.info(f"{col}: {len(df)} total, {skip_count} already normalized (skipped), {len(todo_idx)} pending")

        for idx, i in enumerate(todo_idx, 1):
            raw = str(df.at[i, col])
            df.at[i, std_col] = _llm_normalize_one(raw, prompt)
            if idx % 10 == 0 or idx == len(todo_idx):
                logger.info(f"  {col}: {idx}/{len(todo_idx)} done")
            time.sleep(0.2)   # Light rate limiting to avoid QPS throttling

        df.to_csv(base / f"{col}_unique.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[OK] Completed {col} normalization: {len(todo_idx)} newly processed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    normalize_all(r"consolidated_results.csv")