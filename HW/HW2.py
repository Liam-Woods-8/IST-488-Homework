import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic


def read_url_content(url: str) -> str | None:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


st.title("HW 2 – URL Summarizer")

# URL input at top (required)
url = st.text_input("Web page URL", placeholder="https://example.com")

# Sidebar: summary type (from Lab 2)
summary_type = st.sidebar.selectbox(
    "Summary type",
    (
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ),
)

# Sidebar: output language 
language = st.sidebar.selectbox("Output language", ("English", "French", "Spanish"))

# Sidebar: LLM choice + keep advanced checkbox
llm = st.sidebar.selectbox("LLM", ("OpenAI (GPT)", "Claude (Anthropic)"))
use_advanced_model = st.sidebar.checkbox("Use advanced model")

# Keys from Streamlit secrets (HW requires secrets.toml / secrets)
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
claude_api_key = st.secrets.get("CLAUDE_API_KEY", "")

# Model mapping
openai_model = "gpt-5-chat-latest" if use_advanced_model else "gpt-5-nano"
claude_model = "claude-3-sonnet-20240229" if use_advanced_model else "claude-3-haiku-20240307"


def build_prompt(page_text: str) -> str:
    return (
        f"{summary_type}. Write the summary in {language}.\n\n"
        f"Content from the URL:\n{page_text}"
    )


def validate_openai_key() -> bool:
    try:
        client = OpenAI(api_key=openai_api_key)
        client.models.list()
        return True
    except Exception:
        return False


def validate_claude_key() -> bool:
    try:
        client = anthropic.Anthropic(api_key=claude_api_key)
        # Minimal call to confirm the key works
        client.messages.create(
            model=claude_model,
            max_tokens=1,
            temperature=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except Exception:
        return False


if st.button("Summarize", disabled=not url):
    page_text = read_url_content(url)
    if not page_text:
        st.error("Could not read the URL. Check the link and try again.")
        st.stop()

    prompt = build_prompt(page_text)

    if llm == "OpenAI (GPT)":
        if not openai_api_key or not validate_openai_key():
            st.error("OpenAI key is missing or invalid.")
            st.stop()

        client = OpenAI(api_key=openai_api_key)
        stream = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        st.write_stream(stream)

    else:
        if not claude_api_key or not validate_claude_key():
            st.error("Claude key is missing or invalid.")
            st.stop()

        client = anthropic.Anthropic(api_key=claude_api_key)
        response = client.messages.create(
            model=claude_model,
            max_tokens=600,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        st.markdown(response.content[0].text)
