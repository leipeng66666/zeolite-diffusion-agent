import pandas as pd
import os

def extract_unique(csv_path: str):
    """Generate unique-value intermediate tables for three columns. Checkpoint resume: preserve existing std_* normalization results"""
    base = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in ["zeolite_name", "guest_molecule"]:
        uniq = df[col].dropna().unique()
        new_df = pd.DataFrame({col: uniq})
        unique_path = os.path.join(base, f"{col}_unique.csv")
        std_col = f"std_{col}"
        # If old unique.csv exists, merge previously normalized results
        if os.path.exists(unique_path):
            old_df = pd.read_csv(unique_path, encoding="utf-8-sig")
            if std_col in old_df.columns:
                mapping = dict(zip(old_df[col].astype(str), old_df[std_col].astype(str)))
                new_df[std_col] = new_df[col].astype(str).map(mapping)
        new_df.to_csv(unique_path, index=False, encoding="utf-8-sig")
