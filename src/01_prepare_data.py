"""Step 1: Download the MATH dataset and save it as CSV.

Run from the project root:  python src/01_prepare_data.py
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
OUT = Path("data")
OUT.mkdir(exist_ok=True)

for split in ["train", "test"]:
    parts = [load_dataset("EleutherAI/hendrycks_math", s, split=split).to_pandas()
             for s in SUBJECTS]
    df = pd.concat(parts, ignore_index=True)
    df["level"] = pd.to_numeric(df["level"].str.extract(r"(\d)")[0], errors="coerce")
    df = df.dropna(subset=["level"]).astype({"level": int})
    df[["problem", "type", "level"]].to_csv(OUT / f"{split}.csv", index=False)
    print(f"{split}: {len(df)} problems")
    print(df["type"].value_counts().to_string(), "\n")
