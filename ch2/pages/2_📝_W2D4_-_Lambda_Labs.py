import os
if not os.path.exists("./images"):
    os.chdir("./ch2")

from st_dependencies import *
styling()

def section_home():
    st.markdown("Coming soon!")

func_list = [section_home]

page_list = ["🏠 Home"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
