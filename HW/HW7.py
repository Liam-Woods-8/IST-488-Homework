import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="News Information Bot", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("news.csv")
    df = df.fillna("")
    return df

@st.cache_resource
def build_index(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors

def get_text_column(df):
    possible_columns = ["title", "description", "summary", "content", "article", "text"]
    existing = [col for col in possible_columns if col in df.columns]
    if existing:
        return df[existing].astype(str).agg(" ".join, axis=1)
    return df.astype(str).agg(" ".join, axis=1)

def search_news(query, df, vectorizer, vectors):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, vectors).flatten()
    results = df.copy()
    results["score"] = scores
    return results.sort_values("score", ascending=False)

def rank_interesting(df):
    results = df.copy()

    if "title" in results.columns:
        title_length = results["title"].astype(str).str.len()
    else:
        title_length = 0

    if "description" in results.columns:
        desc_length = results["description"].astype(str).str.len()
    else:
        desc_length = 0

    results["interest_score"] = title_length + desc_length
    return results.sort_values("interest_score", ascending=False)

df = load_data()
df["combined_text"] = get_text_column(df)
vectorizer, vectors = build_index(df["combined_text"])

st.title("News Information Bot")
st.write("This bot only reports on the articles in the uploaded CSV file.")

query = st.text_input("Ask a question about the news")

col1, col2 = st.columns(2)

with col1:
    search_clicked = st.button("Search News")

with col2:
    interesting_clicked = st.button("Find Most Interesting News")

if search_clicked and query:
    results = search_news(query, df, vectorizer, vectors).head(5)
    st.subheader("Top Results")

    for _, row in results.iterrows():
        title = row["title"] if "title" in row else "No title"
        description = row["description"] if "description" in row else "No description"

        st.markdown(f"**Title:** {title}")
        st.write(f"**Description:** {description}")
        st.write(f"**Relevance Score:** {round(row['score'], 3)}")
        st.write("---")

elif search_clicked and not query:
    st.warning("Please enter a question first.")

if interesting_clicked:
    results = rank_interesting(df).head(5)
    st.subheader("Most Interesting News")

    for _, row in results.iterrows():
        title = row["title"] if "title" in row else "No title"
        description = row["description"] if "description" in row else "No description"

        st.markdown(f"**Title:** {title}")
        st.write(f"**Description:** {description}")
        st.write("**Why this is interesting:** It contains more detail than the other articles.")
        st.write("---")