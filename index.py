import streamlit as st

pg = st.navigation([
    st.Page("pages/app.py", title="App page", url_path="/" )
])

pg.run()