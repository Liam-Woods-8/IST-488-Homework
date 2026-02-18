import sys

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from openai import OpenAI


# -----------------------------
# HTML helpers
# -----------------------------
def html_to_text(html: str) -> str:
    """Lightweight HTML -> plain text (no extra libraries)."""
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?s)<.*?>", " ", html)
    html = re.sub(r"\s+", " ", html).strip()
    return html


def two_chunk_split(text: str) -> list[str]:
    """
    HW4 requires TWO mini-documents per source document.

    Method:
    - split text into 2 halves by character length
    - add small overlap around the midpoint to avoid cutting key sentences

    Why:
    - exactly meets the “two chunks per doc” requirement
    - deterministic + simple + fast
    - overlap reduces context loss at boundary
    """
    text = text.strip()
    if not text:
        return []

    overlap = 200
    mid = len(text) // 2
    first = text[: mid + overlap].strip()
    second = text[max(0, mid - overlap) :].strip()
    return [first, second]


# -----------------------------
# Memory buffer (last 5 interactions)
# -----------------------------
def buffer_last_5_interactions(messages):
    """
    Keep system prompt + last 5 interactions (user+assistant pairs = last 10 messages).
    """
    if not messages:
        return messages

    system = []
    if messages[0]["role"] == "system":
        system = [messages[0]]
        rest = messages[1:]
    else:
        rest = messages

    return system + rest[-10:]


# -----------------------------
# Vector DB (HTML RAG)
# -----------------------------
def create_hw4_vectordb(html_folder: str):
    client = chromadb.PersistentClient(path="./chroma_hw4")

    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )

    collection = client.get_or_create_collection(
        name="HW4Collection",
        embedding_function=embed_fn,
    )

    # Create DB only if it doesn't already exist (i.e., collection has docs)
    try:
        if len(collection.get()["ids"]) > 0:
            return collection
    except Exception:
        pass

    folder_path = Path(html_folder)
    if not folder_path.exists():
        st.error(f"HTML folder not found: {folder_path.resolve()}")
        return collection

    for file_path in folder_path.glob("*.html"):
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html)
        chunks = two_chunk_split(text)

        if len(chunks) != 2:
            continue

        ids = [f"{file_path.name}::chunk0", f"{file_path.name}::chunk1"]
        metadatas = [
            {"source": file_path.name, "chunk": 0},
            {"source": file_path.name, "chunk": 1},
        ]
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    return collection


def retrieve_top_chunks(question: str, n_results: int = 4):
    results = st.session_state.HW4_VectorDB.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    sources = [f"{m['source']} (chunk {m['chunk']})" for m in metas]
    return chunks, sources


# -----------------------------
# UI (HW4 page only)
# -----------------------------
st.title("HW 4 — iSchool Student Orgs Chatbot (RAG)")

# Since this file is HW/HW4.py, Data is at HW/Data
HTML_FOLDER = "HW/Data/*.html"

# Optional debug (remove before submitting if you want)
st.caption(f"Loading HTML from: {Path(HTML_FOLDER).resolve()}")
st.caption(f"HTML files found: {len(list(Path(HTML_FOLDER).glob('*.html')))}")

if "HW4_VectorDB" not in st.session_state:
    st.session_state.HW4_VectorDB = create_hw4_vectordb(HTML_FOLDER)

if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [
        {
            "role": "system",
            "content": "You answer questions about iSchool student organizations using only the provided documents."
        }
    ]

# render history
for m in st.session_state.hw4_messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.write(m["content"])

q = st.chat_input("Ask a question about student organizations...")
if q:
    st.session_state.hw4_messages.append({"role": "user", "content": q})
    st.session_state.hw4_messages = buffer_last_5_interactions(st.session_state.hw4_messages)

    chunks, sources = retrieve_top_chunks(q, n_results=4)
    context = "\n\n---\n\n".join(chunks)

    prompt = (
        "Use ONLY the RAG context below to answer.\n"
        "If the answer is not in the RAG context, say you cannot find it.\n\n"
        f"RAG CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{q}\n"
    )

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content

    st.session_state.hw4_messages.append({"role": "assistant", "content": answer})
    st.session_state.hw4_messages = buffer_last_5_interactions(st.session_state.hw4_messages)

    with st.chat_message("assistant"):
        st.write(answer)

    st.write("Sources:")
    for s in sources:
        st.write(f"- {s}")
