"""RAG explainer: retrieve compliance rules + Groq LLM."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "artifacts" / "faiss_index"


def load_vectorstore() -> FAISS:
    load_dotenv(ROOT / ".env", override=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def explain_transaction(transaction_text: str) -> str:
    load_dotenv(ROOT / ".env", override=True)
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set (add to .env).")

    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(transaction_text)
    context = "\n\n".join(d.page_content for d in docs)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=api_key,
    )

    prompt = (
        "You are a senior payments compliance analyst. Use ONLY the context rules below. "
        "If the context is insufficient, say what is missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Transaction:\n{transaction_text}\n\n"
        "Respond with exactly three numbered sections:\n"
        "1. Why this transaction was flagged\n"
        "2. Which compliance rule(s) apply (cite rule numbers/titles from context)\n"
        "3. Recommended next step\n"
    )
    msg = HumanMessage(content=prompt)
    out = llm.invoke([msg])
    return out.content if hasattr(out, "content") else str(out)


def main() -> None:
    sample = (
        "Transaction TXN0012345678: USD 9,999.99 wire on FEDWIRE from sender BIC BOFAUS3NXXX "
        "at 02:15 local time; receiver CITIUS33XXX; flagged as near reporting threshold with "
        "off-hours pattern."
    )
    text = explain_transaction(sample)
    print(text)


if __name__ == "__main__":
    main()
