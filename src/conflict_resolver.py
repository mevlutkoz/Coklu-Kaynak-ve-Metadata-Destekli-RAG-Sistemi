"""Deterministic, field-level conflict resolution.

Priority per field:
  1. JSON with the most recent effective_date
  2. CSV current value
  3. TXT contract default

LLM is never consulted.
"""

from datetime import datetime
from typing import Any


def _parse_date(s: str | None) -> datetime:
    if not s:
        return datetime.min
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return datetime.min


def resolve_fields(raw_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by field_name, pick winner, annotate overrides."""
    by_field: dict[str, list[dict[str, Any]]] = {}
    for f in raw_facts:
        by_field.setdefault(f["field_name"], []).append(f)

    resolved: list[dict[str, Any]] = []
    for field_name, group in by_field.items():
        json_items = [g for g in group if g["source_type"] == "json"]
        csv_items = [g for g in group if g["source_type"] == "csv"]
        txt_items = [g for g in group if g["source_type"] == "txt"]

        losers: list[dict[str, Any]] = []

        if json_items:
            winner = max(
                json_items, key=lambda g: _parse_date(g.get("effective_date"))
            )
            losers = [g for g in group if g is not winner]
        elif csv_items:
            winner = csv_items[0]
            losers = [g for g in group if g is not winner]
        elif txt_items:
            winner = txt_items[0]
        else:
            continue

        overridden_sources = list(
            {g["source_file"] for g in losers}
        )

        resolved.append(
            {
                "field_name": field_name,
                "value": winner["value"],
                "chosen_source": winner["source_type"],
                "source_file": winner["source_file"],
                "effective_date": winner.get("effective_date"),
                "overridden": bool(overridden_sources),
                "overridden_source": overridden_sources,
                "paket": winner.get("paket"),
            }
        )

    return resolved
