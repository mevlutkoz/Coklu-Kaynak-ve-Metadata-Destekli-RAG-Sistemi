"""Field-level fact gathering across CSV + JSON + TXT-default sources."""

from typing import Any

from src.ingestion import csv_parser, json_parser
from src.ingestion.txt_parser import extract_txt_field_facts


def gather_field_facts(paket: str, fields: list[str]) -> list[dict[str, Any]]:
    """Collect all candidate facts for *paket* × *fields* from every source."""
    if not paket or not fields:
        return []

    facts = csv_parser.get_field_facts(paket, fields)
    facts += json_parser.get_field_facts(paket, fields)

    for tf in extract_txt_field_facts():
        if tf["field_name"] in fields:
            facts.append({**tf, "paket": paket})

    return facts
