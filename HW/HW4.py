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


# HTML -> clean text
def html_to_text(html: str) -> str:
    """Remove script/style tags and strip all HTML tags to get plain text."""
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?s)<.*?>", " ", html)
    html = re.sub(r"\s+", " ", html).strip()
    return html


# Chunking method (REQUIRED)
def two_chunk_split(text: str) -> list[str]:
    """
    Chunking method: split each document into exactly TWO chunks.

    Why this method:
    - The assignment requires two mini-documents per HTML file.
    - Splitting into halves keeps it simple and predictable.
    - We include a small overlap so key info near the middle isn't lost
      (helps retrieval when an answer spans the split point).
    """
    text = text.strip()
    if not text:
        return []

    overlap = 200  # character overlap to reduce boundary loss
    mid = len(text) // 2

    chunk1 = text[: mid + overlap].strip()
    chunk2 = text[max(0, mid - overlap):].strip()

    return [chunk1, chunk2]


# Memory buffer (last 5 interactions)
def buffer_last_5_interactions(messages: list[dict]) -> list[dict]:
    """
    Keep system message + last 10 messages (5 user/assistant pairs).
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


# Vector DB creation 
def create_hw4_vectordb(html_dir: Path):
    chroma_client = chromadb.PersistentClient(path="./chroma_hw4")

    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )

    collection = chroma_client.get_or_create_collection(
        name="HW4Collection",
        embedding_function=embed_fn,
    )

    # Only build DB if empty 
    try:
        if collection.count() > 0:
            return collection
    except Exception:
        pass

    html_files = [
        p for p in html_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".html", ".htm", ".xhtml"]
    ]

    if not html_files:
        st.error(f"No HTML files found in: {html_dir}")
        return collection

    progress = st.progress(0)
    total = len(html_files)

    for i, file_path in enumerate(html_files, start=1):
        raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(raw_html)

        # skip super tiny docs 
        if len(text) < 200:
            progress.progress(i / total)
            continue

        chunks = two_chunk_split(text)
        if len(chunks) != 2:
            progress.progress(i / total)
            continue

        # ensure chunks not empty
        if any(len(c) < 50 for c in chunks):
            progress.progress(i / total)
            continue

        ids = [f"{file_path.name}::chunk0", f"{file_path.name}::chunk1"]
        metadatas = [
            {"source": file_path.name, "chunk": 0},
            {"source": file_path.name, "chunk": 1},
        ]

        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        progress.progress(i / total)

    progress.empty()
    return collection


def retrieve_context(collection, question: str, n_results: int = 4):
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    sources = [f"{m['source']} (chunk {m['chunk']})" for m in metas]
    return docs, sources


# Streamlit App
st.title("HW 4 — iSchool Chatbot Using RAG")

BASE_DIR = Path(__file__).resolve().parent
HTML_DIR = BASE_DIR / "data"

if not HTML_DIR.exists():
    st.error("Your data folder was not found. Expected it at HW/data.")
    st.stop()

html_count = len([
    p for p in HTML_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in [".html", ".htm", ".xhtml"]
])

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "hw4_collection" not in st.session_state:
    with st.spinner("Loading vector database..."):
        st.session_state.hw4_collection = create_hw4_vectordb(HTML_DIR)

if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant for Syracuse student organizations. "
                "Use ONLY the provided RAG context to answer. "
                "If the answer is not in the context, say you cannot find it."
            ),
        }
    ]

# display chat history
for m in st.session_state.hw4_messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Ask a question about student organizations...")

if prompt:
    docs, sources = retrieve_context(st.session_state.hw4_collection, prompt, n_results=4)
    context_block = "\n\n---\n\n".join(docs)

    # Add user message, keep buffer
    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    st.session_state.hw4_messages = buffer_last_5_interactions(st.session_state.hw4_messages)

    # Build messages for the LLM:
    # Keep system and Provide context + question as a USER message
    messages_for_llm = []
    if st.session_state.hw4_messages and st.session_state.hw4_messages[0]["role"] == "system":
        messages_for_llm.append(st.session_state.hw4_messages[0])

    # Add recent conversation (excluding system) before the RAG prompt
    for msg in st.session_state.hw4_messages[1:]:
        messages_for_llm.append(msg)

    rag_user_prompt = (
        "Use the following CONTEXT to answer the user's last question.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        "Rules:\n"
        "- If the context does NOT contain the answer, reply: \"I cannot find that in the provided context.\"\n"
        "- Do not use outside knowledge.\n"
    )

    messages_for_llm.append({"role": "user", "content": rag_user_prompt})

    resp = st.session_state.openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages_for_llm,
    )

    answer = resp.choices[0].message.content or ""

st.session_state.hw4_messages.append({"role": "assistant", "content": answer})
st.session_state.hw4_messages = buffer_last_5_interactions(st.session_state.hw4_messages)

with st.chat_message("assistant"):
    st.write(answer)