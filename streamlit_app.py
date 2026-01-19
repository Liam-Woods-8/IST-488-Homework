import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("MY Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com). "
)

# Ask user for their OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
else:

    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # immediatley validate the API key by listing models.
    try:
        client.models.list()
        st.success("API Key validated successfully.")
    except Exception:
        st.error("Invalid OpenAI API key. Please check and try again.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    question = st.text_area(
        "Now ask a question about the document",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)