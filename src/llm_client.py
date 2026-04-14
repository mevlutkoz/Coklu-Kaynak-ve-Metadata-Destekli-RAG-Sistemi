"""Answer generation — receives only resolver output, never raw retrieval."""

import json
from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = """Sen bir şirket müşteri destek asistanısın.
Sana yalnızca iki veri kaynağı verilecek:
  (a) resolved_facts: deterministik conflict resolver tarafından seçilmiş alan-değer çiftleri.
  (b) policy_clauses: ilgili sözleşme maddeleri.

Kurallar:
1. resolved_facts içindeki value değerlerini doğrudan kullan.
2. overridden=true olan alanlar için eski kaynağın yerini aldığını belirt.
3. Cevabın sonuna "--- Kaynaklar ---" bölümü ekle.
   Her kullandığın fact/clause için: dosya adı, tarih (varsa), override notu.
4. Context'te olmayan bilgiyi üretme.
5. Context yetersizse bunu açıkça belirt."""


def generate_answer(
    soru: str,
    resolved_facts: list[dict[str, Any]],
    policy_clauses: list[dict],
) -> str:
    """Produce a natural-language answer from resolved facts + clauses."""
    if not resolved_facts and not policy_clauses:
        return "Bu soruyu yanıtlamak için yeterli bilgi bulunamadı."

    payload = {
        "resolved_facts": resolved_facts,
        "policy_clauses": policy_clauses,
    }

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Soru: {soru}\n\n"
                    f"Veri:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        temperature=0,
    )
    return resp.choices[0].message.content
