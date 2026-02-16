import os
import glob
import re
import streamlit as st
from bs4 import BeautifulSoup

try:
    import chromadb
    CHROMADB_AVAILABLE = True
    CHROMADB_IMPORT_ERROR = None
except Exception as e:
    chromadb = None
    CHROMADB_AVAILABLE = False
    CHROMADB_IMPORT_ERROR = str(e)
from openai import OpenAI

st.set_page_config(page_title="HW4 — iSchool Chatbot Using RAG", layout="centered")
st.title("HW4 — iSchool Chatbot Using RAG")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
HERE = os.path.dirname(__file__)
HTML_DIR = os.path.join(HERE, "data")
# Use an absolute path for the Chroma DB folder
DB_DIR = "/tmp/chroma_hw4_db"
os.makedirs(DB_DIR, exist_ok=True)
COLLECTION_NAME = "ischool_orgs"
TOP_K = 4
OVERLAP_SENTENCES = 2
MAX_CHARS = 12000

def cap_text(s: str, max_chars: int = MAX_CHARS) -> str:
    return s[:max_chars]

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

def build_vector_db_once():
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
        st.stop()

    html_files = sorted(glob.glob(os.path.join(HTML_DIR, "*.html")))
    if not html_files:
        st.error(f"No .html files found in: {HTML_DIR}")
        st.stop()

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    oai = st.session_state.openai_client

    def embed_texts(text_list, batch_size=50):
        all_embs = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            resp = oai.embeddings.create(model="text-embedding-3-small", input=batch)
            all_embs.extend([d.embedding for d in resp.data])
        return all_embs

    batch_texts = []
    batch_meta = []
    batch_ids = []

    for fp in html_files:
        base = os.path.basename(fp)
        text = read_html_as_text(fp)
        c1, c2 = chunk_into_two(text)

        batch_texts.extend([cap_text(c1), cap_text(c2)])
        batch_meta.extend([{"source": base, "chunk": 1}, {"source": base, "chunk": 2}])
        batch_ids.extend([f"{base}::1", f"{base}::2"])

    batch_embs = embed_texts(batch_texts)

    collection.add(
        documents=batch_texts,
        metadatas=batch_meta,
        ids=batch_ids,
        embeddings=batch_embs
    )

def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def retrieve_context(collection, query: str, k: int = TOP_K) -> str:
    oai = st.session_state.openai_client
    q_emb = oai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    results = collection.query(query_embeddings=[q_emb], n_results=k)
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

st.write("DB_DIR:", DB_DIR)
st.write("DB exists now?:", db_exists())
if os.path.isdir(DB_DIR):
    st.write("DB files:", os.listdir(DB_DIR))

if not db_exists():
    if not CHROMADB_AVAILABLE:
        st.error(
            "ChromaDB is not available in this environment.\n"
            f"Import error: {CHROMADB_IMPORT_ERROR}\n"
            "Chroma requires sqlite3 >= 3.35.0; consider upgrading sqlite or running in a different environment."
        )
        st.stop()

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
