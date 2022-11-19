import os
if not os.path.exists("./images"):
    os.chdir("./ch2")

from st_dependencies import *

def section_home():
    st.error("Note - this page will be undergoing significant changes in the future. Its current contents don't represent the ideal vision of the ARENA programme.")
    st.markdown("""
## 1️⃣ Introduction to Docker

In the first half if today, you'll be going through a step-by-step process on how to set up a basic Docker project. This closely resembles the online tutorial from [LogRocket](https://blog.logrocket.com/build-deploy-flask-app-using-docker/).

## 2️⃣

Once you're familiar with the basics of Docker, you'll have a chance to deploy a model inside your app, e.g. your Shakespeare-trained transformer, or your GANs. There are multiple different ways we'll suggest for you to do this, depending on your background with web development, or tools such as Streamlit.""")

def section_1():
    st.markdown("""
## Build and deploy a Flask app using Docker

If you've ever built a web application with Python, chances are that you used a framework to achieve this, one of which could be Flask. Flask is an open-source, beginner-friendly web framework built on the Python programming language. Flask is suitable when you want to develop an application with a light codebase rapidly.

Docker is an open-source tool that enables you to containerize your applications. It aids in building, testing, deploying, and managing your applications within an isolated environment, and we’ll use it to do everything except test in this article.

## Getting started

### Docker

You'll have to install Docker. You can find the instructions specific to your OS here:

* [Install docker desktop on mac](https://docs.docker.com/desktop/install/mac-install/)
* [Install docker desktop on linux](https://docs.docker.com/desktop/install/linux-install/)
* [Install docker desktop on windows](https://docs.docker.com/desktop/install/windows-install/)

The installation might involve jumping through a few hoops depending on which OS you have, so you might need to do some Googling of the error messages you get. For instance, as a Windows user, I was required to install and upgrade **WSL** (Windows Subsystem for Linux).

### Heroku

Heroku is a cloud platform where developers can build and run applications in the cloud. If you don't already have an account with Heroku, you should create one [here](https://signup.heroku.com/).

### Misc.

You should have at least Python version 3.8 installed on your machine.

Although not strictly necessary, it would help if you had some familiarity with command line interfaces. If you haven't yet gone through the prerequisite material 
""")

def section_2():
    pass

def section_empty():
    st.markdown("""Coming soon!

In the meantime, the following readings might help:

* [Guide to getting started with Docker](https://docs.docker.com/get-started/) \*
* [Best practices from Docker documentation](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [Video: Docker best practices](https://www.youtube.com/watch?v=8vXoMqWgbQQ) (18 mins)
* [Build and deploy a Flask app using Docker](https://blog.logrocket.com/build-deploy-flask-app-using-docker/) (the exercises will approximately follow the structure of this tutorial)
""")

func_list = [section_home, section_1, section_2]

page_list = ["🏠 Home", "1️⃣ Introduction to Docker", "2️⃣ Deploying an app"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:

        radio = st.radio("Section", page_list)

        st.markdown("---")

    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
