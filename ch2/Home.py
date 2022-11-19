from st_dependencies import *
styling()

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
    st_image("headers/scale.png", width=320)

    st.markdown("""# Training at scale

There are a number of techniques that are helpful for training large-scale models efficiently. Here, we will learn more about these techniques and how to use them.

The purpose of this week isn't just to give you a theoretical overview of the topics, but to have you come away from this week with concrete skills that will be important for research engineers. So the focus won't just be on reading, but also on hands-on use of the tools we'll discuss during this section.

Some highlights from this chapter include:

* Building docker containers to house the models you've built
* Using Lambda Labs to access cloud compute
* Learning about & using tools from distributed computing
""")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if is_local or check_password():
    page()
