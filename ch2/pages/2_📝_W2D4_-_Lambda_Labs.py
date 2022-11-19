import os
if not os.path.exists("./images"):
    os.chdir("./ch2")

from st_dependencies import *

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
