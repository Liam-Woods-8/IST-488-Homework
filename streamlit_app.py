import streamlit as st

st.title("HW Manager")

hw1_page = st.Page("HW/HW1.py", title="HW 1")
hw2_page = st.Page("HW/HW2.py", title="HW 2")
hw3_page = st.Page("HW/HW3.py", title="HW 3")

pg = st.navigation([hw3_page, hw2_page, hw1_page])  
pg.run()