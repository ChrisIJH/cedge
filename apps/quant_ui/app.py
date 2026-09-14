"""
apps/quant_ui/app.py


Entry point. Streamlit's multipage navigation reads pages/ automatically;
this file only sets shared page config and a landing view.

Run:
    streamlit run apps/quant_ui/app.py
"""
import streamlit as st
from lib import api

st.set_page_config(page_title="CEdge", layout="wide")

st.title("CEdge")
st.caption("A quant investment research platform")

st.subheader("Service Status")

cols = st.columns(len(api.SERVICE_URLS))

for col, (name, url) in zip(cols, api.SERVICE_URLS.items()):
    with col:
        if api.healthy(name):
            st.success(f"**{name}**\n\n{url!r}")
        else:
            st.error(f"**{name}**\n\n{url!r}\n\nnot reachable")

st.divider()
st.markdown("Select a page from the sidebar.")