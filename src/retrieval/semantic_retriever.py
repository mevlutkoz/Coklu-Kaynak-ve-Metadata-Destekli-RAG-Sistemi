"""Hybrid (semantic + keyword) retrieval over madde-chunked contract text."""

import chromadb
from openai import OpenAI

COLLECTION_NAME = "sozlesme"


def _keyword_score(text: str, keywords: list[str]) -> int:
    tl = text.lower()
    return sum(1 for kw in keywords if kw.lower() in tl)


def search_policy_clauses(soru: str, top_k: int = 3) -> list[dict]:
    """Return the top-k contract clauses relevant to *soru*."""
    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path="vectorstore/")
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        return []

    emb = openai_client.embeddings.create(
        model="text-embedding-3-small", input=soru
    ).data[0].embedding

    fetch_k = min(top_k * 2, collection.count())
    results = collection.query(query_embeddings=[emb], n_results=fetch_k)

    keywords = soru.lower().split()
    clauses: list[dict] = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        semantic = 1 - results["distances"][0][i]
        kw = _keyword_score(doc, keywords)
        score = semantic + kw * 0.1
        clauses.append(
            {
                "text": doc,
                "madde": meta.get("madde", ""),
                "source_file": "sozlesme.txt",
                "_score": score,
            }
        )

    clauses.sort(key=lambda x: x["_score"], reverse=True)
    for c in clauses:
        del c["_score"]
    return clauses[:top_k]
