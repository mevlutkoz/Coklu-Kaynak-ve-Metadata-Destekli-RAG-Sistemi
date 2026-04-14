"""Top-level orchestration: plan → retrieve → resolve → answer."""

from typing import Any

from src.fields import ALL_FIELDS
from src.ingestion.txt_parser import ingest_txt
from src.query_planner import plan_query
from src.retrieval.semantic_retriever import search_policy_clauses
from src.retrieval.structured_retriever import gather_field_facts
from src.conflict_resolver import resolve_fields
from src.llm_client import generate_answer


def _expand_fields(plan: dict[str, Any]) -> list[str]:
    """Determine the full set of fields to retrieve based on the plan."""
    fields = list(plan.get("asked_fields") or [])

    if plan.get("asks_current_package_info"):
        for f in ALL_FIELDS:
            if f not in fields:
                fields.append(f)

    # iptal / iade policy soruları → iade_suresi_gun de gerekebilir
    if plan.get("asks_contract_policy") and "iade_suresi_gun" not in fields:
        fields.append("iade_suresi_gun")

    return fields


def run(soru: str) -> dict[str, Any]:
    """Run the full RAG pipeline for a single user question."""
    ingest_txt()

    plan = plan_query(soru)
    print(f"Query plan: {plan}")

    # --- Field-level retrieval + resolution ---
    raw_facts: list[dict[str, Any]] = []
    if plan.get("paket"):
        fields = _expand_fields(plan)
        if fields:
            raw_facts = gather_field_facts(plan["paket"], fields)

    resolved = resolve_fields(raw_facts)

    # --- Policy clause retrieval (TXT) ---
    policy_clauses: list[dict] = []
    if plan.get("asks_contract_policy"):
        policy_clauses = search_policy_clauses(soru)

    print(
        f"{len(raw_facts)} ham fact → {len(resolved)} resolved, "
        f"{len(policy_clauses)} policy clause"
    )

    cevap = generate_answer(soru, resolved, policy_clauses)

    return {
        "cevap": cevap,
        "resolved_facts": resolved,
        "policy_clauses": policy_clauses,
        "query_plan": plan,
    }
