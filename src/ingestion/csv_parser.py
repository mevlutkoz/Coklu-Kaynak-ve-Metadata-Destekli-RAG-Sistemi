"""CSV ingestion — fresh pd.read_csv on every call, never cached."""

from typing import Any

import pandas as pd

DATA_PATH = "data/paket_fiyatlari.csv"


def load_csv(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the package-price table from disk."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df


def get_field_facts(
    paket: str, fields: list[str], path: str = DATA_PATH
) -> list[dict[str, Any]]:
    """Return one fact dict per requested field for *paket*."""
    df = load_csv(path)
    filtre = df[df["paket"].str.lower() == paket.lower()]
    if filtre.empty:
        return []
    row = filtre.iloc[0]
    facts: list[dict[str, Any]] = []
    for f in fields:
        if f in row.index:
            val = row[f]
            # pandas int64/float64 → native Python for JSON serialisation
            if hasattr(val, "item"):
                val = val.item()
            facts.append(
                {
                    "field_name": f,
                    "value": val,
                    "source_type": "csv",
                    "source_file": "paket_fiyatlari.csv",
                    "effective_date": None,
                    "paket": paket,
                }
            )
    return facts
