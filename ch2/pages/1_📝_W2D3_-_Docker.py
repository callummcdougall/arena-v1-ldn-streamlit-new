import streamlit as st
import base64
st.set_page_config(layout="wide")
import os
if os.path.exists(os.getcwd() + "/images"):
    is_local = True
else:
    is_local = False
    os.chdir("./ch2")
st.write(is_local)

def img_to_html(img_path, width):
    with open("images/" + img_path, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    return f"<img style='width:{width}px;max-width:100%;margin-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"

st.markdown("""
<style>
label.effi0qh3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 15px;
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
    color: red;
    white-space: pre-wrap !important;
}
code:not(h1 code):not(h2 code):not(h3 code):not(h4 code) {
    font-size: 13px;
}
a.contents-el > code {
    color: black;
    background-color: rgb(248, 249, 251);
}
.css-ffhzg2 a.contents-el > code {
    color: orange;
    background-color: rgb(26, 28, 36);
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

def section_home():
    st.error("Note - this page will be undergoing significant changes in the future. Its current contents don't represent the ideal vision of the ARENA programme.")
    st.markdown("""
## 1️⃣ Introduction to Docker

Today, you'll be going through a tutorial on how to set up a basic Docker project. In the first half of tomorrow, we'll extend this by having you deploy a model inside this app (e.g. your Shakespeare-trained transformer, or your GANs).""")


def section_docker():
    st.error("Note - this page will be undergoing significant changes in the future. Its current contents don't represent the ideal vision of the ARENA programme.")
    st.markdown("""
## Reading

Below are a selection of different resources. You should read at least one of the ones with an asterisk \*, although it's up to you which ones you want to read. You can always come back to some of these later.

* [Guide to getting started with Docker](https://docs.docker.com/get-started/) \*
* [Best practices from Docker documentation](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [Video: Docker best practices](https://www.youtube.com/watch?v=8vXoMqWgbQQ) (18 mins)

You might also want to try [Play with Docker](https://labs.play-with-docker.com/), which is a project allowing users to run Docker commands simply in your browser.

Make sure you have a basic understanding the following topics before moving on:

* What are docker containers and images
* Why you might want to use docker
* What is the docker daemon

You will also have to use **Flask** and **Heroku** during this tutorial. Don't worry if you haven't come across them before, since you don't need to have deep knowledge about either of them. However, it might be worth reading up a little on both of these once you get to those sections - for instance, at the links [here](https://trifinlabs.com/what-is-heroku/) and [here](https://pymbook.readthedocs.io/en/latest/flask.html#:~:text=Flask%20is%20a%20web%20framework,application%20or%20a%20commercial%20website.).

## Introduction to Docker

You should follow [this tutorial](https://blog.logrocket.com/build-deploy-flask-app-using-docker/). By the end, you should have a working container spun up (although it won't contain anything interesting yet!).

### Getting started

First, you'll need to install Docker. You can find the instructions specific to your OS here:

* [Install docker desktop on mac](https://docs.docker.com/desktop/install/mac-install/)
* [Install docker desktop on linux](https://docs.docker.com/desktop/install/linux-install/)
* [Install docker desktop on windows](https://docs.docker.com/desktop/install/windows-install/)

The installation might involve jumping through a few hoops depending on which OS you have, so you might need to do some Googling of the error messages you get.

### Tips

* The tutorial [says](https://blog.logrocket.com/build-deploy-flask-app-using-docker/#:~:text=Your%20requirements.txt%20file%20should%20contain%20at%20least%20the%20following%20content%3A) that your `requirements.txt` file should contain at least a set of 8 libraries, which it helpfully lists. If your `pip install -r requirements.txt` command is failing, then you can remove all libraries from `requirements.txt` and replace it with these - then it's much more likely to work.
* When you use the command line interface, your Docker username will always be written in lowercase, even if you used upper case when signing up.

## Bonus exercises

Here are some exercises for once you get to the end of the tutorial.

### Pull someone else's image

Pull an image which someone else in the class pushed to dockerhub: run it on your local machine.

Enter the CLI of this new container: 
* What processes is it running?
* How much memory is it using?

Update your Dockerfile to have an argument to specify the environment variable NREPEATS. Update your app to return some text (any text) repeated NREPEATS times over
* Build and run the new image with NREPEATS set to 5: visit it in the browser to check it works
* When might specifying arguments at the point of running the image be useful?

### Deploy a model inside your app

We'll look at this more tomorrow (and more detailed instructions will be provided). You might find [this tutorial](https://www.photoroom.com/tech/packaging-pytorch-in-docker/) helpful.

## Extra reading

* [Get started with Dockerhub](https://docs.docker.com/docker-hub/)
* [On mounting volumes to containers](https://docs.docker.com/storage/volumes/)
* [15 minute intro to Kubernetes](https://www.youtube.com/watch?v=VnvRFRk_51k)
""")

def section_empty():
    st.markdown("""Coming soon!

In the meantime, the following readings might help:

* [Guide to getting started with Docker](https://docs.docker.com/get-started/) \*
* [Best practices from Docker documentation](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [Video: Docker best practices](https://www.youtube.com/watch?v=8vXoMqWgbQQ) (18 mins)
* [Build and deploy a Flask app using Docker](https://blog.logrocket.com/build-deploy-flask-app-using-docker/) (the exercises will approximately follow the structure of this tutorial)
""")

func_list = [section_home, section_docker]

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
