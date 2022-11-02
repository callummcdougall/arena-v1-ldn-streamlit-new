import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<style>
button[data-baseweb="tab"] {
    font-family: monospace;
}
p {
    line-height:1.48em;
}
.streamlit-expanderHeader {
    font-size: 1em;
    color: darkblue;
}
.css-ffhzg2 .streamlit-expanderHeader {
    color: lightblue;
}
header {
    background: rgba(255, 255, 255, 0) !important;
}
code {
    color:red;
    white-space: pre-wrap !important;
}
.css-ffhzg2 code:not(pre code) {
    color: orange;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
pre code {
    font-size:13px !important;
}
.katex {
    font-size:17px;
}
h2 .katex, h3 .katex, h4 .katex {
    font-size: unset;
}
ul.contents {
    line-height:1.3em; 
    list-style:none;
    color-black;
    margin-left: -10px;
}
ul.contents a, ul.contents a:link, ul.contents a:visited, ul.contents a:active {
    color: black;
    text-decoration: none;
}
ul.contents a:hover {
    color: black;
    text-decoration: underline;
}
</style>""", unsafe_allow_html=True)

with st.sidebar:

    tabs = st.tabs([f"w1d{i}" for i in range(1, 6)])

    for tab in tabs:
        with tab:
            st.markdown("""
            ## Table of Contents

            <ul class="contents">
                <li><a class="contents-el" href="#reading">Reading</a></li>
                <li><a class="contents-el" href="#einops">Einops</a></li>
                <li><a class="contents-el" href="#einsum">Einsum</a></li>
            </ul>
            """, unsafe_allow_html=True)