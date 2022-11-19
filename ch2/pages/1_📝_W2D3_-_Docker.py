import os
if not os.path.exists("./images"):
    os.chdir("./ch2")

from st_dependencies import *

def section_home():
    st.error("Note - this page will be undergoing significant changes in the future. Its current contents don't represent the ideal vision of the ARENA programme.")
    st.markdown("""
## 1️⃣ Introduction to Docker

Today, you'll be going through a tutorial on how to set up a basic Docker project. In the first half of tomorrow, we'll extend this by having you deploy a model inside this app (e.g. your Shakespeare-trained transformer, or your GANs).""")


def section_1():
    pass

def section_empty():
    st.markdown("""Coming soon!

In the meantime, the following readings might help:

* [Guide to getting started with Docker](https://docs.docker.com/get-started/) \*
* [Best practices from Docker documentation](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [Video: Docker best practices](https://www.youtube.com/watch?v=8vXoMqWgbQQ) (18 mins)
* [Build and deploy a Flask app using Docker](https://blog.logrocket.com/build-deploy-flask-app-using-docker/) (the exercises will approximately follow the structure of this tutorial)
""")

func_list = [section_home, section_1]

page_list = ["🏠 Home", "1️⃣ Docker"]
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
