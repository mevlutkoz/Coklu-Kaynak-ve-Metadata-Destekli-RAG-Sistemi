"""Case-hardening tests for the field-level RAG pipeline.

Tests 1-7 are pure logic — no OpenAI calls needed.
Test 8 (query planner) requires a live API key.
"""

import json
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.ingestion import csv_parser, json_parser
from src.ingestion.txt_parser import _compute_hash, ingest_txt, needs_reindex, HASH_PATH, DATA_PATH
from src.retrieval.structured_retriever import gather_field_facts
from src.conflict_resolver import resolve_fields


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def temp_csv(tmp_path: Path) -> str:
    p = tmp_path / "paket.csv"
    p.write_text(
        "paket,fiyat_tl,kullanici_limiti,depolama_gb,destek_turu,iade_suresi_gun\n"
        "Pro,299,25,100,öncelikli destek,14\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture()
def temp_json(tmp_path: Path) -> str:
    p = tmp_path / "upd.json"
    p.write_text(
        json.dumps(
            [
                {
                    "tarih": "2024-06-01",
                    "etkilenen_paket": "Pro",
                    "degisiklik": "Pro paket iade süresi 30 güne çıkarıldı.",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return str(p)


# ── 1. CSV fresh read ────────────────────────────────────────────────────

def test_csv_loader_reads_fresh_every_time(temp_csv: str) -> None:
    df1 = csv_parser.load_csv(temp_csv)
    assert int(df1.iloc[0]["fiyat_tl"]) == 299

    Path(temp_csv).write_text(
        "paket,fiyat_tl,kullanici_limiti,depolama_gb,destek_turu,iade_suresi_gun\n"
        "Pro,399,25,100,öncelikli destek,14\n",
        encoding="utf-8",
    )
    df2 = csv_parser.load_csv(temp_csv)
    assert int(df2.iloc[0]["fiyat_tl"]) == 399


# ── 2. JSON fresh read ──────────────────────────────────────────────────

def test_json_loader_reads_fresh_every_time(temp_json: str) -> None:
    u1 = json_parser.load_updates(temp_json)
    assert len(u1) == 1

    data = u1 + [
        {
            "tarih": "2025-01-01",
            "etkilenen_paket": "Pro",
            "degisiklik": "Pro paket fiyatı 399 TL'ye güncellendi.",
        }
    ]
    Path(temp_json).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    u2 = json_parser.load_updates(temp_json)
    assert len(u2) == 2


# ── 3. JSON override beats CSV ──────────────────────────────────────────

def test_json_override_beats_csv_for_same_field() -> None:
    raw = [
        {
            "field_name": "iade_suresi_gun",
            "value": 14,
            "source_type": "csv",
            "source_file": "paket_fiyatlari.csv",
            "effective_date": None,
            "paket": "Pro",
        },
        {
            "field_name": "iade_suresi_gun",
            "value": 30,
            "source_type": "json",
            "source_file": "guncellemeler.json",
            "effective_date": "2024-06-01",
            "paket": "Pro",
        },
    ]
    res = resolve_fields(raw)
    assert len(res) == 1
    r = res[0]
    assert r["value"] == 30
    assert r["chosen_source"] == "json"
    assert r["overridden"] is True
    assert "paket_fiyatlari.csv" in r["overridden_source"]


# ── 4. CSV beats TXT ────────────────────────────────────────────────────

def test_csv_beats_txt_when_no_json_override() -> None:
    raw = [
        {
            "field_name": "iade_suresi_gun",
            "value": 14,
            "source_type": "txt",
            "source_file": "sozlesme.txt",
            "effective_date": None,
        },
        {
            "field_name": "iade_suresi_gun",
            "value": 14,
            "source_type": "csv",
            "source_file": "paket_fiyatlari.csv",
            "effective_date": None,
            "paket": "Pro",
        },
    ]
    res = resolve_fields(raw)
    assert len(res) == 1
    assert res[0]["chosen_source"] == "csv"
    assert res[0]["overridden"] is True
    assert "sozlesme.txt" in res[0]["overridden_source"]


# ── 5. Mixed sources ────────────────────────────────────────────────────

def test_mixed_answer_uses_multiple_sources() -> None:
    raw = [
        {
            "field_name": "fiyat_tl",
            "value": 299,
            "source_type": "csv",
            "source_file": "paket_fiyatlari.csv",
            "effective_date": None,
            "paket": "Pro",
        },
        {
            "field_name": "iade_suresi_gun",
            "value": 14,
            "source_type": "csv",
            "source_file": "paket_fiyatlari.csv",
            "effective_date": None,
            "paket": "Pro",
        },
        {
            "field_name": "iade_suresi_gun",
            "value": 30,
            "source_type": "json",
            "source_file": "guncellemeler.json",
            "effective_date": "2024-06-01",
            "paket": "Pro",
        },
    ]
    res = resolve_fields(raw)
    d = {r["field_name"]: r for r in res}
    assert d["fiyat_tl"]["value"] == 299
    assert d["fiyat_tl"]["chosen_source"] == "csv"
    assert d["iade_suresi_gun"]["value"] == 30
    assert d["iade_suresi_gun"]["chosen_source"] == "json"


# ── 6. Dynamic JSON update ──────────────────────────────────────────────

def test_new_json_record_changes_answer_without_restart() -> None:
    path = "data/guncellemeler.json"
    backup = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(backup)
        data.append(
            {
                "tarih": "2030-01-01",
                "etkilenen_paket": "Pro",
                "degisiklik": "Pro paket iade süresi 90 güne çıkarıldı.",
            }
        )
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        facts = json_parser.get_field_facts("Pro", ["iade_suresi_gun"])
        resolved = resolve_fields(facts)
        r = next(f for f in resolved if f["field_name"] == "iade_suresi_gun")
        assert r["value"] == 90
        assert r["effective_date"] == "2030-01-01"
    finally:
        Path(path).write_text(backup, encoding="utf-8")


# ── 7. Contract reindex guard ───────────────────────────────────────────

def test_contract_reindex_only_when_txt_changes() -> None:
    Path(HASH_PATH).parent.mkdir(exist_ok=True)
    Path(HASH_PATH).write_text(_compute_hash(DATA_PATH))
    assert needs_reindex() is False
    assert ingest_txt() == 0


# ── 8. Query planner shape (requires OpenAI) ────────────────────────────

def test_query_planner_structured_output_shape() -> None:
    from src.query_planner import plan_query

    plan = plan_query("Pro paket fiyatı nedir ve iade süresi kaç gün?")
    assert set(plan.keys()) == {
        "paket",
        "asked_fields",
        "asks_contract_policy",
        "asks_current_package_info",
    }
    assert plan["paket"] == "Pro"
    assert isinstance(plan["asked_fields"], list)
    assert "fiyat_tl" in plan["asked_fields"]
    assert "iade_suresi_gun" in plan["asked_fields"]
    assert isinstance(plan["asks_contract_policy"], bool)
    assert isinstance(plan["asks_current_package_info"], bool)


# ── 9. JSON field extraction covers all 5 canonical fields ─────────────

@pytest.mark.parametrize("text,expected_field,expected_value", [
    ("Pro paket fiyatı 299 TL'ye güncellendi.", "fiyat_tl", 299),
    ("Pro paket iade süresi 30 güne çıkarıldı.", "iade_suresi_gun", 30),
    ("Pro paket kullanıcı limiti 25'e yükseltildi.", "kullanici_limiti", 25),
    ("Enterprise paket depolama kapasitesi 1000 GB'a yükseltildi.", "depolama_gb", 1000),
    ("Enterprise paket destek türü premium destek olarak güncellendi.", "destek_turu", "premium destek"),
])
def test_extract_field_covers_all_five_types(text: str, expected_field: str, expected_value) -> None:
    result = json_parser.extract_field_from_update(text)
    assert result is not None, f"Failed to extract from: {text}"
    field, value = result
    assert field == expected_field
    assert value == expected_value


# ── 10. Most recent JSON wins for same field ───────────────────────────

def test_most_recent_json_wins_when_multiple_updates_exist() -> None:
    raw = [
        {
            "field_name": "kullanici_limiti",
            "value": 25,
            "source_type": "json",
            "source_file": "guncellemeler.json",
            "effective_date": "2024-03-01",
            "paket": "Pro",
        },
        {
            "field_name": "kullanici_limiti",
            "value": 30,
            "source_type": "json",
            "source_file": "guncellemeler.json",
            "effective_date": "2025-03-01",
            "paket": "Pro",
        },
        {
            "field_name": "kullanici_limiti",
            "value": 25,
            "source_type": "csv",
            "source_file": "paket_fiyatlari.csv",
            "effective_date": None,
            "paket": "Pro",
        },
    ]
    res = resolve_fields(raw)
    assert len(res) == 1
    r = res[0]
    assert r["value"] == 30
    assert r["chosen_source"] == "json"
    assert r["effective_date"] == "2025-03-01"
    assert r["overridden"] is True


# ── 11. Non-numeric CSV value preserved ────────────────────────────────

def test_non_numeric_csv_value_preserved(temp_csv: str) -> None:
    Path(temp_csv).write_text(
        "paket,fiyat_tl,kullanici_limiti,depolama_gb,destek_turu,iade_suresi_gun\n"
        "Enterprise,999,sınırsız,1000,7/24 destek,30\n",
        encoding="utf-8",
    )
    facts = csv_parser.get_field_facts("Enterprise", ["kullanici_limiti"], temp_csv)
    assert len(facts) == 1
    assert facts[0]["value"] == "sınırsız"
    assert facts[0]["field_name"] == "kullanici_limiti"


# ── 12. All 5 fields resolved from mixed sources ──────────────────────

def test_all_five_fields_resolved_from_mixed_sources() -> None:
    raw = [
        {"field_name": "fiyat_tl", "value": 299, "source_type": "csv",
         "source_file": "paket_fiyatlari.csv", "effective_date": None, "paket": "Pro"},
        {"field_name": "iade_suresi_gun", "value": 14, "source_type": "csv",
         "source_file": "paket_fiyatlari.csv", "effective_date": None, "paket": "Pro"},
        {"field_name": "iade_suresi_gun", "value": 30, "source_type": "json",
         "source_file": "guncellemeler.json", "effective_date": "2024-06-01", "paket": "Pro"},
        {"field_name": "depolama_gb", "value": 100, "source_type": "csv",
         "source_file": "paket_fiyatlari.csv", "effective_date": None, "paket": "Pro"},
        {"field_name": "depolama_gb", "value": 200, "source_type": "json",
         "source_file": "guncellemeler.json", "effective_date": "2024-07-10", "paket": "Pro"},
        {"field_name": "destek_turu", "value": "öncelikli destek", "source_type": "csv",
         "source_file": "paket_fiyatlari.csv", "effective_date": None, "paket": "Pro"},
        {"field_name": "kullanici_limiti", "value": 10, "source_type": "txt",
         "source_file": "sozlesme.txt", "effective_date": None, "paket": "Pro"},
    ]
    res = resolve_fields(raw)
    d = {r["field_name"]: r for r in res}
    assert len(d) == 5
    assert d["fiyat_tl"]["chosen_source"] == "csv"
    assert d["iade_suresi_gun"]["chosen_source"] == "json"
    assert d["iade_suresi_gun"]["value"] == 30
    assert d["depolama_gb"]["chosen_source"] == "json"
    assert d["depolama_gb"]["value"] == 200
    assert d["destek_turu"]["chosen_source"] == "csv"
    assert d["kullanici_limiti"]["chosen_source"] == "txt"


# ── 13. destek_turu JSON overrides CSV ─────────────────────────────────

def test_destek_turu_json_overrides_csv() -> None:
    raw = [
        {"field_name": "destek_turu", "value": "7/24 destek", "source_type": "csv",
         "source_file": "paket_fiyatlari.csv", "effective_date": None, "paket": "Enterprise"},
        {"field_name": "destek_turu", "value": "premium destek", "source_type": "json",
         "source_file": "guncellemeler.json", "effective_date": "2025-01-10", "paket": "Enterprise"},
    ]
    res = resolve_fields(raw)
    assert len(res) == 1
    assert res[0]["value"] == "premium destek"
    assert res[0]["chosen_source"] == "json"
    assert res[0]["overridden"] is True
    assert "paket_fiyatlari.csv" in res[0]["overridden_source"]


# ── 14. Empty / missing input edge cases ───────────────────────────────

def test_empty_inputs_return_no_facts() -> None:
    assert gather_field_facts("Pro", []) == []
    assert gather_field_facts("", ["fiyat_tl"]) == []
    assert resolve_fields([]) == []


# ── 15. Unrecognized update text returns None ──────────────────────────

def test_unrecognized_update_text_returns_none() -> None:
    assert json_parser.extract_field_from_update("Bu bir genel duyurudur.") is None
    assert json_parser.extract_field_from_update("Sistem bakım çalışması yapılacaktır.") is None
