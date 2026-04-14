"""TXT (sözleşme) ingestion — madde-based chunking and embedding.

Hash-based reindex guard: only re-embed when the file content changes.
Also extracts base/default field values from known maddelar.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI

DATA_PATH = "data/sozlesme.txt"
HASH_PATH = "vectorstore/sozlesme_hash.txt"
COLLECTION_NAME = "sozlesme"


def _compute_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def needs_reindex(
    data_path: str = DATA_PATH, hash_path: str = HASH_PATH
) -> bool:
    """Return True when the TXT file differs from the saved hash."""
    if not Path(hash_path).exists():
        return True
    with open(hash_path) as f:
        return f.read().strip() != _compute_hash(data_path)


def chunk_by_madde(text: str) -> list[dict[str, str]]:
    """Split contract text into madde-keyed chunks."""
    pattern = r"(Madde \d+\.\d+:)"
    parts = re.split(pattern, text)
    chunks: list[dict[str, str]] = []
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body = parts[i + 1].strip()
        m = re.search(r"Madde (\d+\.\d+)", header)
        if m:
            chunks.append({"text": f"{header} {body}", "madde": m.group(1)})
        i += 2
    return chunks


def extract_txt_field_facts(data_path: str = DATA_PATH) -> list[dict[str, Any]]:
    """Extract base/default field values from well-known contract maddelar."""
    with open(data_path, encoding="utf-8") as f:
        text = f.read()

    facts: list[dict[str, Any]] = []
    m = re.search(
        r"Madde 4\.1:.*?(\d+)\s*(?:\([^)]*\))?\s*g[üu]n", text, re.DOTALL
    )
    if m:
        facts.append(
            {
                "field_name": "iade_suresi_gun",
                "value": int(m.group(1)),
                "source_type": "txt",
                "source_file": "sozlesme.txt",
                "effective_date": None,
                "madde": "4.1",
            }
        )
    return facts


def ingest_txt(force: bool = False) -> int:
    """Embed madde-chunks into ChromaDB. Returns count; 0 if skipped."""
    current_hash = _compute_hash(DATA_PATH)

    if not force and Path(HASH_PATH).exists():
        with open(HASH_PATH) as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            print("sozlesme.txt değişmemiş, re-index atlanıyor.")
            return 0

    client = chromadb.PersistentClient(path="vectorstore/")
    openai_client = OpenAI()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_by_madde(text)
    for chunk in chunks:
        emb = openai_client.embeddings.create(
            model="text-embedding-3-small", input=chunk["text"]
        ).data[0].embedding
        collection.add(
            documents=[chunk["text"]],
            embeddings=[emb],
            metadatas=[
                {"kaynak": "sozlesme.txt", "madde": chunk["madde"], "tarih": ""}
            ],
            ids=[f"madde_{chunk['madde'].replace('.', '_')}"],
        )

    Path(HASH_PATH).parent.mkdir(exist_ok=True)
    with open(HASH_PATH, "w") as f:
        f.write(current_hash)

    print(f"{len(chunks)} madde embed edildi ve ChromaDB'ye yüklendi.")
    return len(chunks)
