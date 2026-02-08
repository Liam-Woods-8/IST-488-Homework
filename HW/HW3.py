import streamlit as st
from openai import OpenAI
import anthropic


st.title("HW3 — A Streaming Chatbot that Discusses a URL")

st.write(
    "This chatbot answers questions using up to two URLs you provide in the sidebar. "
    "It streams the assistant response as it is generated. "
    "Conversation memory: a buffer of 6 messages (3 user–assistant exchanges). "
    "The system prompt (including URL context) is never discarded."
)

# Define the options (put these in the sidebar):
url1 = st.sidebar.text_input("URL 1 (optional)")
url2 = st.sidebar.text_input("URL 2 (optional)")

llm_choice = st.sidebar.selectbox(
    "LLM",
    ("OpenAI (GPT-5)", "Anthropic (Claude)"),
)

load_urls = st.sidebar.button("Load URL(s)")


# the read_url_content() function from HW2
def read_url_content(url: str) -> str:
    # paste your HW2 version here (exactly)
    # it should return cleaned page text as a string
    raise NotImplementedError("Paste your HW2 read_url_content(url) here.")


SYSTEM_BASE = (
    "You are a helpful chatbot. Use the URL text below as your main source. "
    "If the answer is not in the URL text, say you are not sure. "
    "Explain things so a 10-year-old can understand."
)

def build_system_prompt(url_text: str) -> dict:
    return {"role": "system", "content": f"{SYSTEM_BASE}\n\nURL TEXT:\n{url_text}"}


def buffer_6_messages_keep_system(messages):
    if not messages:
        return messages

    system = []
    rest = messages

    if messages[0]["role"] == "system":
        system = [messages[0]]
        rest = messages[1:]

    rest = rest[-6:]
    return system + rest


openai_api_key = st.secrets["OPENAI_API_KEY"]
claude_api_key = st.secrets["CLAUDE_API_KEY"]

if "url_text" not in st.session_state:
    st.session_state.url_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = [build_system_prompt("")]

if load_urls:
    texts = []
    if url1.strip():
        texts.append(read_url_content(url1.strip()))
    if url2.strip():
        texts.append(read_url_content(url2.strip()))

    st.session_state.url_text = "\n\n---\n\n".join([t for t in texts if t]).strip()
    st.session_state.messages = [build_system_prompt(st.session_state.url_text)]


# Display chat history 
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Ask a question about the URL(s)...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages = buffer_6_messages_keep_system(st.session_state.messages)

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

        system_text = st.session_state.messages[0]["content"] if st.session_state.messages and st.session_state.messages[0]["role"] == "system" else ""
        convo = [m for m in st.session_state.messages if m["role"] != "system"]

        with st.chat_message("assistant"):
            collected = []
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=700,
                temperature=0.2,
                system=system_text,
                messages=convo,
            ) as stream:
                for text in stream.text_stream:
                    collected.append(text)
                    st.write(text, end="")

            assistant_text = "".join(collected)

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    st.session_state.messages = buffer_6_messages_keep_system(st.session_state.messages)
