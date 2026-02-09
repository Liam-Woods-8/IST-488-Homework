import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic


st.title("HW 03 — A Streaming Chatbot that Discusses a URL")

st.write(
    "This chatbot answers questions using one or two URLs you enter in the sidebar. "
    "It streams responses while generating them. "
    "Conversation memory: a buffer of 6 messages (3 user–assistant exchanges). "
    "The system prompt (with URL context) is never discarded."
)


def read_url_content(url: str) -> str | None:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


def build_system_message(url_text: str) -> dict:
    system_prompt = (
        "You are a helpful chatbot. Use the URL text below as your main source. "
        "If the answer is not in the URL text, say you are not sure. "
        "Explain things so a 10-year-old can understand."
        "\n\nURL TEXT:\n"
        f"{url_text}"
    )
    return {"role": "system", "content": system_prompt}


def buffer_6_keep_system(messages):
    if not messages:
        return messages

    system = []
    rest = messages

    if messages[0]["role"] == "system":
        system = [messages[0]]
        rest = messages[1:]

    rest = rest[-6:]
    return system + rest


openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
claude_api_key = st.secrets.get("CLAUDE_API_KEY", "")

url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com")
url2 = st.sidebar.text_input("URL 2 (optional)", placeholder="https://example.com")

llm_choice = st.sidebar.selectbox(
    "LLM",
    ("OpenAI (GPT-5)", "Claude (Anthropic Opus)"),
)

use_both = st.sidebar.checkbox("Use both URLs as context", value=True)
load_urls = st.sidebar.button("Load URL(s)")


if "url_text" not in st.session_state:
    st.session_state.url_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = [build_system_message("")]


if load_urls:
    texts = []

    if url1.strip():
        t1 = read_url_content(url1.strip())
        if t1:
            texts.append(t1)

    if use_both and url2.strip():
        t2 = read_url_content(url2.strip())
        if t2:
            texts.append(t2)

    st.session_state.url_text = "\n\n---\n\n".join(texts).strip()
    st.session_state.messages = [build_system_message(st.session_state.url_text)]


for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Ask a question about the URL(s)...")
if prompt:
    if not st.session_state.url_text:
        st.info("Load at least one URL first using the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages = buffer_6_keep_system(st.session_state.messages)

    if llm_choice == "OpenAI (GPT-5)":
        client = OpenAI(api_key=openai_api_key)

        stream = client.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=st.session_state.messages,
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        client = anthropic.Anthropic(api_key=claude_api_key)

        system_text = st.session_state.messages[0]["content"]
        convo = [m for m in st.session_state.messages if m["role"] != "system"]

        with st.chat_message("assistant"):
            collected = []
            placeholder = st.empty()
            full = ""

            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=700,
                temperature=0.2,
                system=system_text,
                messages=convo,
            ) as stream:
                for text in stream.text_stream:
                    collected.append(text)
                    full += text
                    placeholder.markdown(full)

            assistant_text = "".join(collected)

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    st.session_state.messages = buffer_6_keep_system(st.session_state.messages)
