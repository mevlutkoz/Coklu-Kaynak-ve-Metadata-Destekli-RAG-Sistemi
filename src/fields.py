"""Canonical field names shared across CSV, JSON, and conflict resolver."""

FIELD_FIYAT = "fiyat_tl"
FIELD_LIMIT = "kullanici_limiti"
FIELD_DEPOLAMA = "depolama_gb"
FIELD_DESTEK = "destek_turu"
FIELD_IADE = "iade_suresi_gun"

ALL_FIELDS: list[str] = [
    FIELD_FIYAT,
    FIELD_LIMIT,
    FIELD_DEPOLAMA,
    FIELD_DESTEK,
    FIELD_IADE,
]
