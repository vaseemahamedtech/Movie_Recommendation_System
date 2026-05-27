"""
Exploratory Data Analysis — prints summary statistics & distributions.
"""
import pandas as pd
import numpy as np

def run_eda(df: pd.DataFrame):
    print("\n── EDA SUMMARY ──────────────────────────────────────────")
    print(f"  Shape          : {df.shape}")
    print(f"  Columns        : {list(df.columns)}")
    print(f"\n  Rating Stats:")
    print(df["rating"].describe().to_string())

    print(f"\n  Top 5 Genres:")
    print(df["primary_genre"].value_counts().head(5).to_string())

    print(f"\n  Movies per Decade:")
    print(df["decade"].value_counts().sort_index().to_string())

    print(f"\n  Avg Rating by Genre:")
    avg = df.groupby("primary_genre")["rating"].mean().sort_values(ascending=False).round(2)
    print(avg.to_string())

    corr = df[["rating","votes","runtime_min","revenue_usd","budget_usd"]].corr().round(3)
    print(f"\n  Correlation Matrix:")
    print(corr.to_string())
    print("─" * 60)
