import os
import glob
import re
import streamlit as st
from bs4 import BeautifulSoup

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

st.set_page_config(page_title="HW4 — iSchool Chatbot Using RAG", layout="centered")
st.title("HW4 — iSchool Chatbot Using RAG")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
HTML_DIR = "/workspaces/Homework-1-IST488/HW/data"
DB_DIR = "chroma_hw4_db"
COLLECTION_NAME = "ischool_orgs"
TOP_K = 4
OVERLAP_SENTENCES = 2

def read_html_as_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


# Chunking method:
# The HTML text is first split into sentences and then divided into two halves.
# A small overlap of sentences is kept between the halves so important ideas are
# not cut in the middle. This method is simple, readable, and ensures exactly
# two balanced chunks per document as required by the assignment.
def chunk_into_two(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        mid = max(1, len(text) // 2)
        return [text[:mid].strip(), text[mid:].strip()]

    mid = len(sentences) // 2
    a_end = min(len(sentences), mid + OVERLAP_SENTENCES)
    b_start = max(0, mid - OVERLAP_SENTENCES)

    chunk_a = " ".join(sentences[:a_end]).strip()
    chunk_b = " ".join(sentences[b_start:]).strip()

    if not chunk_a:
        chunk_a = " ".join(sentences[:mid]).strip()
    if not chunk_b:
        chunk_b = " ".join(sentences[mid:]).strip()

    return [chunk_a, chunk_b]

def db_exists() -> bool:
    return os.path.isdir(DB_DIR) and len(os.listdir(DB_DIR)) > 0

def get_embedding_fn():
    return OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )

def build_vector_db_once():
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
        st.stop()

    html_files = sorted(glob.glob(os.path.join(HTML_DIR, "*.html")))
    if not html_files:
        st.error(f"No .html files found in: {HTML_DIR}")
        st.stop()

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_fn()
    )

    docs, metas, ids = [], [], []

    for fp in html_files:
        base = os.path.basename(fp)
        text = read_html_as_text(fp)
        c1, c2 = chunk_into_two(text)

        docs.extend([c1, c2])
        metas.extend(
            [{"source": base, "chunk": 1}, {"source": base, "chunk": 2}]
        )
        ids.extend([f"{base}::1", f"{base}::2"])

    collection.add(documents=docs, metadatas=metas, ids=ids)

def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_fn()
    )

def retrieve_context(collection, query: str, k: int = TOP_K) -> str:
    results = collection.query(query_texts=[query], n_results=k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    blocks = []
    for d, m in zip(docs, metas):
        src = m.get("source", "unknown")
        ch = m.get("chunk", "?")
        blocks.append(f"[{src} | chunk {ch}] {d}")

    return "\n\n".join(blocks).strip()

def keep_last_5_interactions(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    system = [messages[0]] if messages[0]["role"] == "system" else []
    rest = messages[1:] if system else messages
    rest = rest[-10:]
    return system + rest

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)

if not db_exists():
    st.info("Vector DB not found. Building it once from the HTML files...")
    build_vector_db_once()

if "collection" not in st.session_state:
    st.session_state.collection = get_collection()

def base_system_message() -> dict:
    return {
        "role": "system",
        "content": (
            "You answer questions about iSchool student organizations.\n"
            "Use ONLY the CONTEXT provided.\n"
            "If the answer is not in the context, say: \"I’m not sure based on the documents I have.\""
        )
    }

if "messages" not in st.session_state:
    st.session_state.messages = [base_system_message()]

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about the student orgs...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages = keep_last_5_interactions(st.session_state.messages)

    ctx = retrieve_context(st.session_state.collection, prompt, TOP_K)
    context_system = {
        "role": "system",
        "content": f"CONTEXT:\n{ctx}"
    }

    base_system = st.session_state.messages[0]
    convo = st.session_state.messages[1:]

    stream = st.session_state.openai_client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=[base_system, context_system] + convo,
        stream=True,
    )

    with st.chat_message("assistant"):
        answer = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.messages = keep_last_5_interactions(st.session_state.messages)
