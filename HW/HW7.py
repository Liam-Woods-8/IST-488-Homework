import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="News Information Bot", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("news.csv")
    return df.fillna("")

def prepare_data(df):
    df = df.copy()
    df["combined_text"] = (
        df["company_name"].astype(str) + " " +
        df["Document"].astype(str)
    )
    df["parsed_date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

@st.cache_resource
def build_index(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors

def search_news(query, df, vectorizer, vectors):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, vectors).flatten()
    results = df.copy()
    results["score"] = scores
    return results.sort_values("score", ascending=False)

def rank_interesting(df):
    results = df.copy()

    text_score = results["Document"].astype(str).str.len()

    if "days_since_2000" in results.columns:
        recency_score = pd.to_numeric(results["days_since_2000"], errors="coerce").fillna(0)
    else:
        recency_score = 0

    results["interest_score"] = text_score + (recency_score * 0.05)
    return results.sort_values("interest_score", ascending=False)

def display_search_results(results):
    for _, row in results.iterrows():
        st.markdown(f"**Company:** {row['company_name']}")
        st.write(f"**Date:** {row['Date']}")
        st.write(f"**Article:** {row['Document']}")
        st.write(f"**URL:** {row['URL']}")
        st.write(f"**Relevance Score:** {round(row['score'], 3)}")
        st.write("---")

def display_interesting_results(results):
    for _, row in results.iterrows():
        st.markdown(f"**Company:** {row['company_name']}")
        st.write(f"**Date:** {row['Date']}")
        st.write(f"**Article:** {row['Document']}")
        st.write(f"**URL:** {row['URL']}")
        st.write(f"**Interest Score:** {round(row['interest_score'], 3)}")
        st.write("**Why this is interesting:** It is more detailed and/or more recent than many of the other articles.")
        st.write("---")

df = load_data()
df = prepare_data(df)
vectorizer, vectors = build_index(df["combined_text"])

st.title("News Information Bot")
st.write("This bot only reports on the articles in the uploaded CSV file.")

query = st.text_input("Ask a question about the news")

col1, col2 = st.columns(2)

with col1:
    search_clicked = st.button("Search News")

with col2:
    interesting_clicked = st.button("Find Most Interesting News")

if search_clicked:
    if query.strip():
        results = search_news(query, df, vectorizer, vectors).head(5)
        st.subheader("Top Results")
        display_search_results(results)
    else:
        st.warning("Please enter a question first.")

if interesting_clicked:
    results = rank_interesting(df).head(5)
    st.subheader("Most Interesting News")
    display_interesting_results(results)