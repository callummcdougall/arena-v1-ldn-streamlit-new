import os
if not os.path.exists("./images"):
    os.chdir("./ch5")

from st_dependencies import *
st.set_page_config(layout="wide")

st.markdown("""
<style>
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

# st.sidebar.markdown("""
# ## Table of Contents

# <ul class="contents">
#     <li><a class="contents-el" href="#about-this-page">About this page</a></li>
#     <li><a class="contents-el" href="#hints">Hints</a></li>
#     <li><a class="contents-el" href="#test-functions">Test functions</a></li>
#     <li><a class="contents-el" href="#tips">Tips</a></li>
#     <li><a class="contents-el" href="#support">Support</a></li>
# </ul>
# """, unsafe_allow_html=True)

def page():
    st_image("headers/mod.png", width=320)

    st.markdown("""# Modelling objectives

This chapter takes a tour through various generative models. This is the name for a broad class of models which can generate new data instances from a particular distribution. Examples are diffusion models like [DALL-E 2](https://openai.com/dall-e-2/), which we used to generate the images you’re seeing on this website!

Some highlights from this chapter include:

* Building a GAN to generate celebrity faces
* Building a variational autoencoder for MNIST
* Building and running a diffusion model
""")

if is_local or check_password():
    page()
