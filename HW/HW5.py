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


# Chunking method 
def two_chunk_split(text: str) -> list[str]:
    """
    Chunking method: split each document into exactly TWO chunks.

    Why this method:
    - The assignment requires two mini-documents per HTML file.
    - Splitting into halves keeps it simple and predictable.
    - We include a small overlap so key info near the middle isn't lost.
    """
    text = text.strip()
    if not text:
        return []

    overlap = 200
    mid = len(text) // 2

    chunk1 = text[: mid + overlap].strip()
    chunk2 = text[max(0, mid - overlap):].strip()

    return [chunk1, chunk2]


# Memory buffer (last 5 interactions)
def buffer_last_5_interactions(messages: list[dict]) -> list[dict]:
    """Keep system message + last 10 messages (5 user/assistant pairs)."""
    if not messages:
        return messages

    system = []
    if messages[0]["role"] == "system":
        system = [messages[0]]
        rest = messages[1:]
    else:
        rest = messages

    return system + rest[-10:]


# Vector DB creation (same as HW4)
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


# HW5 : retrieval function (LLM query -> Chroma -> formatted context)
def relevant_club_info(query: str, n_results: int = 4) -> str:
    """
    Takes input 'query' (from the LLM) and returns relevant information
    from the ChromaDB collection as a formatted context block.
    """
    query = (query or "").strip()
    if not query:
        return "No query provided."

    collection = st.session_state.hw5_collection

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []

    if not docs:
        return "No relevant context found in the provided documents."

    blocks = []
    for i, doc in enumerate(docs):
        src = ""
        if metas and i < len(metas) and metas[i]:
            src = f"{metas[i].get('source', '')} (chunk {metas[i].get('chunk', '')})".strip()
        label = f"Source: {src}" if src else f"Chunk {i+1}"
        blocks.append(f"{label}\n{doc}")

    return "\n\n---\n\n".join(blocks)

# Make the query come "from the LLM" (simple rewrite step)
def make_retrieval_query(user_question: str) -> str:
    """
    Produces a short search query (1 line) for vector retrieval.
    This satisfies HW5's 'query (from the LLM)' requirement.
    """
    prompt = (
        "Rewrite the user's question into a short search query to retrieve "
        "relevant passages from the student organizations documents. "
        "Keep it under 20 words.\n\n"
        f"User question: {user_question}\n"
        "Search query:"
    )

    resp = st.session_state.openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip().strip('"')



# Streamlit App
st.title("HW 5 — Enhance Your Chatbot (Intelligent RAG)")

BASE_DIR = Path(__file__).resolve().parent
HTML_DIR = BASE_DIR / "data"

if not HTML_DIR.exists():
    st.error("Your data folder was not found. Expected it at HW/data.")
    st.stop()

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "hw5_collection" not in st.session_state:
    with st.spinner("Loading vector database..."):
        st.session_state.hw5_collection = create_hw4_vectordb(HTML_DIR)

if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant for Syracuse student organizations. "
                "Use ONLY the provided document context to answer. "
                "If the answer is not in the context, say you cannot find it."
            ),
        }
    ]

# Display chat history
for m in st.session_state.hw5_messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_prompt = st.chat_input("Ask a question about student organizations...")

if user_prompt:
    # Show user message
    st.session_state.hw5_messages.append({"role": "user", "content": user_prompt})
    st.session_state.hw5_messages = buffer_last_5_interactions(st.session_state.hw5_messages)

    with st.chat_message("user"):
        st.write(user_prompt)

    # 1) LLM generates the retrieval query
    retrieval_query = make_retrieval_query(user_prompt)

    # 2) Retrieval function does vector search
    context_block = relevant_club_info(retrieval_query, n_results=4)

    # 3) Invoke LLM with memory + retrieved context (no function calling)
    messages_for_llm = []

    # Keep system
    if st.session_state.hw5_messages and st.session_state.hw5_messages[0]["role"] == "system":
        messages_for_llm.append(st.session_state.hw5_messages[0])

    # Add recent conversation
    for msg in st.session_state.hw5_messages[1:]:
        messages_for_llm.append(msg)

    # Provide retrieved context explicitly (system OR user message both ok)
    messages_for_llm.append(
        {
            "role": "system",
            "content": (
                "Retrieved context (use this to answer):\n\n"
                f"{context_block}\n\n"
                'If the answer is not in the context, reply: "I cannot find that in the provided context."'
            ),
        }
    )

    resp = st.session_state.openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages_for_llm,
    )

    answer = resp.choices[0].message.content or ""

    st.session_state.hw5_messages.append({"role": "assistant", "content": answer})
    st.session_state.hw5_messages = buffer_last_5_interactions(st.session_state.hw5_messages)

    with st.chat_message("assistant"):
        st.write(answer)