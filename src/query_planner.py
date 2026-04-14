"""LLM-based intent extraction with JSON schema structured output.

Returns a structured plan; never makes decisions — that is the resolver's job.
"""

import json
from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = """Sen bir intent extraction asistanısın.
Kullanıcının sorusunu analiz et ve aşağıdaki JSON formatında yanıt ver.

Alanlar:
- paket: Soruda bir paket adı geçiyorsa "Basic"|"Pro"|"Enterprise", yoksa null.
- asked_fields: Soruda sorulan yapısal alanlar. Geçerli değerler:
  * fiyat_tl           (ücret, fiyat, TL, ne kadar, maliyet)
  * kullanici_limiti   (kullanıcı sayısı, limit, kişi)
  * depolama_gb        (depolama, GB, alan, kapasite)
  * destek_turu        (destek, müşteri hizmetleri, yardım)
  * iade_suresi_gun    (iade süresi, kaç gün iade, geri alma süresi)
- asks_contract_policy: true eğer soru sözleşme maddesi, iptal koşulu,
  ödeme politikası, iade prosedürü, hak veya yükümlülük içeriyorsa.
  "iptal edersem", "iade nasıl yapılır", "koşullar nelerdir" gibi.
- asks_current_package_info: true eğer soru paketin genel bilgisini soruyorsa
  ("Pro paketi nedir", "Enterprise paketi hakkında bilgi").

Sadece JSON döndür."""

SCHEMA: dict[str, Any] = {
    "name": "query_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "paket": {
                "type": ["string", "null"],
                "enum": ["Basic", "Pro", "Enterprise", None],
            },
            "asked_fields": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "fiyat_tl",
                        "kullanici_limiti",
                        "depolama_gb",
                        "destek_turu",
                        "iade_suresi_gun",
                    ],
                },
            },
            "asks_contract_policy": {"type": "boolean"},
            "asks_current_package_info": {"type": "boolean"},
        },
        "required": [
            "paket",
            "asked_fields",
            "asks_contract_policy",
            "asks_current_package_info",
        ],
        "additionalProperties": False,
    },
}


def plan_query(soru: str) -> dict[str, Any]:
    """Extract structured intent from the user question via gpt-4o-mini."""
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": soru},
        ],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": SCHEMA},
    )
    return json.loads(resp.choices[0].message.content)
