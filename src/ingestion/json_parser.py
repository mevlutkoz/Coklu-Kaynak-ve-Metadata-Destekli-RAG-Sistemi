"""JSON update-log ingestion with field-level extraction helpers."""

import json
import re
from typing import Any

DATA_PATH = "data/guncellemeler.json"
REQUIRED_FIELDS = {"tarih", "etkilenen_paket", "degisiklik"}


def load_updates(path: str = DATA_PATH) -> list[dict[str, str]]:
    """Load update records fresh from disk, validating required keys."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for i, kayit in enumerate(data):
        eksik = REQUIRED_FIELDS - set(kayit.keys())
        if eksik:
            raise ValueError(f"Kayıt {i} eksik alan içeriyor: {eksik}")
    return data


# ---------------------------------------------------------------------------
# Field extraction from free-text Turkish change descriptions
# ---------------------------------------------------------------------------

def extract_field_from_update(text: str) -> tuple[str, Any] | None:
    """Parse a change sentence into (field_name, new_value).

    Returns None when no known field pattern matches.
    """
    t = text.lower()

    if "iade süresi" in t or "iade suresi" in t:
        m = re.findall(r"(\d+)\s*g[üu]n", t)
        if m:
            return ("iade_suresi_gun", int(m[-1]))

    if "kullanıcı limiti" in t or "kullanici limiti" in t:
        nums = re.findall(r"(\d+)", t)
        if nums:
            return ("kullanici_limiti", int(nums[-1]))

    if "depolama" in t or "kapasite" in t:
        m = re.findall(r"(\d+)\s*gb", t)
        if m:
            return ("depolama_gb", int(m[-1]))

    if "fiyat" in t:
        m = re.findall(r"(\d+)\s*tl", t)
        if m:
            return ("fiyat_tl", int(m[-1]))

    if "destek türü" in t or "destek turu" in t:
        for label in ("7/24 destek", "öncelikli destek", "email destek", "premium destek"):
            if label in t:
                return ("destek_turu", label)

    return None


def get_field_facts(
    paket: str, fields: list[str], path: str = DATA_PATH
) -> list[dict[str, Any]]:
    """Return normalized JSON facts matching *paket* and *fields*."""
    updates = load_updates(path)
    facts: list[dict[str, Any]] = []
    for u in updates:
        if u["etkilenen_paket"].lower() != paket.lower():
            continue
        parsed = extract_field_from_update(u["degisiklik"])
        if parsed is None:
            continue
        field_name, value = parsed
        if field_name not in fields:
            continue
        facts.append(
            {
                "field_name": field_name,
                "value": value,
                "source_type": "json",
                "source_file": "guncellemeler.json",
                "effective_date": u["tarih"],
                "paket": paket,
                "raw_text": u["degisiklik"],
            }
        )
    return facts
