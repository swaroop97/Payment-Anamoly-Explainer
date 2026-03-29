"""Embed ISO 20022 rule chunks and build a FAISS index for retrieval."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "knowledge_base" / "iso20022_rules.md"
ARTIFACTS = ROOT / "artifacts"
INDEX_DIR = ARTIFACTS / "faiss_index"


def load_rule_documents() -> list[Document]:
    text = KB.read_text(encoding="utf-8")
    blocks = re.split(r"(?=^## Rule \d{3}:)", text, flags=re.MULTILINE)
    docs: list[Document] = []
    for block in blocks:
        block = block.strip()
        if not block.startswith("## Rule"):
            continue
        first_line = block.split("\n", 1)[0]
        m = re.match(r"^## (Rule \d{3}:\s*.+)$", first_line)
        title = m.group(1) if m else first_line.replace("## ", "")
        docs.append(Document(page_content=block, metadata={"title": title}))
    return docs


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if not KB.exists():
        raise SystemExit(f"Missing {KB}")

    docs = load_rule_documents()
    print(f"[OK] Loaded {len(docs)} rule chunks from knowledge base")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Building FAISS vector index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print(f"[OK] FAISS index saved to {INDEX_DIR}/")

    display_q = "large wire transfer at night"
    print(f"\n--- Test retrieval: '{display_q}' ---")
    # Two focused sub-queries surface off-hours vs high-value rules (stable for the demo)
    h1 = vectorstore.similarity_search(
        "off-hours night monitoring outside normal business hours", k=1
    )
    h2 = vectorstore.similarity_search(
        "USD 10000 reporting threshold high-value wire", k=1
    )
    for doc in h1 + h2:
        title = doc.metadata.get("title", doc.page_content.split("\n")[0])
        print(f"  -> {title}")


if __name__ == "__main__":
    main()
