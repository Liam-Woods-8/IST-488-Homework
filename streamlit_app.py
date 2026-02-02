import streamlit as st

st.title("HW Manager")

hw1_page = st.Page("HW/HW1.py", title="HW 1")
hw2_page = st.Page("HW/HW2.py", title="HW 2")

pg = st.navigation([hw2_page, hw1_page])  # default to HW2 first (optional)
pg.run()