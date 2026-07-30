# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

def map_back(csv_path: str) -> None:
    base = Path(csv_path).parent
    main = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in ["zeolite_name", "guest_molecule"]:
        map_df = pd.read_csv(base / f"{col}_unique.csv", encoding="utf-8-sig")
        d = dict(zip(map_df[col], map_df[f"std_{col}"]))
        main[f"std_{col}"] = main[col].map(d)
    main.to_csv(csv_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    map_back(r"consolidated_results.csv")