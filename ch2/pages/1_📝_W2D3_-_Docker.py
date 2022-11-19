import os
if not os.path.exists("./images"):
    os.chdir("./ch2")

from st_dependencies import *
styling()

def section_home():
    st.markdown("""
## 1️⃣ Introduction to Docker

In the first half if today, you'll be going through a step-by-step process on how to set up a basic Docker project. This closely resembles the online tutorial from [LogRocket](https://blog.logrocket.com/build-deploy-flask-app-using-docker/).

## 2️⃣ Deploying a PyTorch model

Once you're familiar with the basics of Docker, you'll have a chance to deploy a model inside your app, e.g. your Shakespeare-trained transformer, or your GANs. There are multiple different ways we'll suggest for you to do this, depending on your background with web development, or tools such as Streamlit.""")

def section_1():
    st.markdown("""
## Introduction to Docker

Docker is an open platform for developing, shipping, and running applications.

There are two key parts of the Docker infrastructure which are vital to understand: **containers** and **keys**.

**Docker Containers*** are sandboxed processes on your machine, isolated from all other processes on the host machine. It is portable, can be run on local or virtual machines, deployed to the cloud, etc. You can think of them as similar to virtual machines, except that they are significantly more streamlined operating environments; providing only the resources an application actually needs to function. Containers are created by the `docker run` command.

**Docker Images** are read-only files containg the source code, libraries, dependencies, tools, and other files needed for an application to run. They are called images because they represent a snapshot of an application and its environment at a specific point in time. This is particularly helpful because it provides uniformity and consistency - developers can modify and test software in stable, unchanging conditions.

Images can exist without containers, but the reverse is not true. A container can be seen as a living instance of a Docker image.

A few more important concepts to cover:

* The **Docker Registry** is a cataloging system for hosting, pushing and pulling Docker images. These can be local, or third party services. In this tutorial, we'll be using the official Docker Registry for this, also known as [Docker Hub](https://docs.docker.com/registry/).
* The **Docker daemon**, called `dockerd`, listens for Docker API requests and manages Docker objects such as images and containers. When you run commands using the Docker client `docker` (e.g. `docker build`), the Docker daemon bridges the gap between the client and the rest of the Docker architecture.""")

    st_image("docker_architecture.png", 500)

    st.markdown("""
Don't worry if this seems a little abstract right now - as we get more hands-on throughout the rest of this tutorial, this should all make a lot more sense.

## Build and deploy a Flask app using Docker

If you've ever built a web application with Python, chances are that you used a framework to achieve this, one of which could be Flask. Flask is an open-source, beginner-friendly web framework built on the Python programming language. Flask is suitable when you want to develop an application with a light codebase rapidly.

Docker is an open-source tool that enables you to containerize your applications. It aids in building, testing, deploying, and managing your applications within an isolated environment, and we'll use it to do everything except test in this article.

## Getting started

### Docker

You'll have to install Docker. You can find the instructions specific to your OS here:

* [Install docker desktop on mac](https://docs.docker.com/desktop/install/mac-install/)
* [Install docker desktop on linux](https://docs.docker.com/desktop/install/linux-install/)
* [Install docker desktop on windows](https://docs.docker.com/desktop/install/windows-install/)

The installation might involve jumping through a few hoops depending on which OS you have, so you might need to do some Googling of the error messages you get. For instance, as a Windows user, I was required to install and upgrade **WSL** (Windows Subsystem for Linux).

You'll also need to create an account for Docker, including a username. 

### Heroku

Heroku is a platform where developers can build and run applications in the cloud. If you don't already have an account with Heroku, you should create one [here](https://signup.heroku.com/).

You will also need to install Heroku CLI, which is the interface you will be using to run Heroku commands. A set of instructions for doing this can be found [here](https://www.geeksforgeeks.org/introduction-and-installation-of-heroku-cli-on-windows-machine/).

### Misc.

You should have at least Python version 3.8 installed on your machine.

You will find it helpful to have some basic familiarity with command line interfaces. If you haven't yet gone through the corresponding prerequisite material, you can do so [here](https://arena-prerequisites.streamlit.app/#unix). At the very least, you should understand what CLIs are, and how to use commands like `cd`, `cp`, `mkdir`.

## Creating the Flask app

If you haven't come across Flask before, you can read [this short introduction](https://www.geeksforgeeks.org/python-introduction-to-web-development-using-flask/) to get an idea of how it works. You should follow up to at least the section on **using variables in Flask**, however if you want to create interesting Flask applications later on then you are strongly recommended to read the whole page.

One you've read this and feel comfortable with the basics of Flask, you should create a simple Flask application that renders a message on the browser. Create a folder with the name `flask_docker` to contain your application.

```
mkdir flask_docker
```

Next, `cd` into the `flask_docker` directory and run the below command to install Flask (possibly swapping out `pip` for whatever package installer you use).

```
pip install Flask
```

After successfully installing Flask, the next step is to create a Python file that receives and responds to requests in our application. Create a `view.py` file that will contain the Python code snippet below:

```python
from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
```

The purpose of most of these lines should be clear from the Flask introduction, but we'll run through it again here. 

The `@app.route` annotation serves to direct the request to a mapped URL. In this case, the provided URL is `/`, which represents the homepage.

This annotation also has a `method` parameter that takes a list of HTTP methods to specify the permitted method for the provided URL. By default (as illustrated), the GET method is the only permitted HTTP method.

Here is an example of how you can specify that your route should permit both `GET` and `POST` HTTP methods:

```python
@app.route('/', methods=['POST', 'GET'])
```

See [here](https://www.w3schools.com/tags/ref_httpmethods.asp) for an explanation of the difference between `GET` and `POST`.

The `home()` function bound to the URL provided in the @app.route annotation will run when you send a `GET` request to this route. The function returns a call to `render_template` that in turn renders the content of the `index.html` file, which we will create in the next section.

```python
port = int(os.environ.get('PORT', 5000))
app.run(debug=True, host='0.0.0.0', port=port)
```

The above portion of the `view.py` file is required when we deploy this application to Heroku, which we will demonstrate in the subsequent section. Not including this will cause your application to crash on Heroku.

## The HTML template

The next step is to create the `index.html` file and provide the content we want to render on the browser when you invoke the `home()` function in the `view.py` file.

Within the root directory, create a `templates` directory, then create the `index.html` file. Add the code snippet below to the HTML file:

```html
<!DOCTYPE html>

<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flask Docker</title>
</head>
<body>
    <h1>This is a Flask App containerised with Docker</h1>
</body>
</html>
```

## Writing Python requirement files with Docker

If you've ever explored any published Python project, you may have noticed a `requirements.txt` file. This file contains the list of packages and dependencies that you need to run your project and their respective versions.

Within the root directory, run the below command in the terminal:

```
pip freeze > requirements.txt
```

This will generate the names of the packages and their respective versions that you have installed, as well as some other inbuilt dependencies that run your Flask application. Then, it stores them in a `.txt` file named `requirements`.

Depending on the complexity of your project and the packages you have installed, the content of this file will vary from project to project. For instance, when your app involves generating output from PyTorch models, you might need to add `torch` to your requirements.

You can also install the packages contained in this file in another project by copying the `requirements.txt` file to your desired project and running the following command:

```
pip install -r requirements.txt
```

The advantage of doing this is that you don't have to run the pip install command for each package repeatedly.

Your `requirements.txt` file should contain at least the following content:

```python
click==8.0.3
colorama==0.4.4
Flask==2.0.2
itsdangerous==2.0.1
Jinja2==3.0.3
MarkupSafe==2.0.1
Werkzeug==2.0.2
gunicorn==20.1.0
```

The version numbers generated in the requirements file may be different from what is written here because, again, this depends on the type of application you're building and the versions of the packages you have installed when building your app.

A nice thing about containerizing with Docker is that you get to package your application with all of the runtime dependencies that are required to make it self-sufficient. Therefore, your application runs without you needing to worry about incompatibilities with its host environment.

You should now have the following directory structure:

```
flask_docker
  ├ templates
  | └ sub1b
  ├ requirements.txt
  └ view.py
```

You can test that the application works before you proceed to containerize it. Run this command on your terminal within the root directory to perform this test:

```
python view.py
```

This command serves your Flask app. It should give you a url e.g. `http://10.21.85.144:5000/`, which you can use to run your app locally. You should see a page like this:""")

    st_image("flask_app.png", 400)
    st.markdown("""

In general, running your app locally is the best way to see the effect of the changes you make, before you push it to Heroku (which we'll get to later).

## Setting up the Dockerfile

You should have already installed Docker from the link earlier. Now, create a file and name it `Dockerfile`. There should be no extension (e.g. if your OS prompts you to name it `Dockerfile.dockerfile` then refuse, just leave it as `Dockerfile`). Add the following code:

```docker
# start by pulling the python image
FROM python:3.8-alpine

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["view.py" ]
```

Let's go over the instructions in this Dockerfile:

* **`FROM python:3.8-alpine`**
    * Since Docker allows us to inherit existing images, we install a Python image and install it in our Docker image. Alpine is a lightweight Linux distro that will serve as the OS on which we install our image.
* **`COPY ./requirements.txt /app/requirements.txt`**
    * Here, we copy the requirements file and its content (the generated packages and dependencies) into the app folder of the image
    * Remember, the second path here (the one containing `app`) refers to your container - it has no relation to the `flask_docker` directory on the machine you're currently working on.
* **`WORKDIR /app`**
    * We proceed to set the working directory as `/app`, which will be the root directory of our application in the container.
    * Any subsequent commands (e.g. `COPY`, `ENTRYPOINT`, `CMD`) will refer to this directory.
* **`RUN pip install -r requirements.txt`**
    This command installs all the dependencies defined in the `requirements.txt file` into our application within the container.
* **`COPY . /app`**
    * This copies every other file and its respective contents into the `app` folder that is the root directory of our application within the container.
* **`ENTRYPOINT [ "python" ]`**
    * This is the command that runs the application in the container.
* **`CMD [ "view.py" ]`**
    * Finally, this appends the list of parameters to the EntryPoint parameter to perform the command that runs the application. This is similar to how you would run the Python app on your terminal using the `python view.py` command.
    * The difference between `CMD` and `ENTRYPOINT` is as follows:
        * `ENTRYPOINT` specifies a command that will **always** be executed when the container starts.
        * `CMD` instructions are fed into `ENTRYPOINT`; they can be overridden from the command line when you run docker via `docker run`.

## Build the Docker image

Let's proceed to build the image with the command below:

```
docker image build -t flask_docker .
```

You can also add the flag `--progress=plain` to the end of this command, if you want to see logs during the building process.

## Run the container

After successfully building the image, the next step is to run an instance of the image. Here is how to perform this:

```
docker run -p 5000:5000 -d flask_docker
```

This command runs the container and its embedded application, each on port 5000 using a port-binding approach. The first 5000 is the port that we allocate to the container on our machine. The second 5000 is the port where the application will run on the container.

Once you see this command, you should see the same webpage as earlier when you send the request `localhost:5000` in your browser.

## Deploying your Flask app to Docker Hub

As we mentioned above, Docker Hub is a registry where Docker users can create, test, and manage containers. If you’ve worked with GitHub, this section will be very familiar to you.

Follow the next sequence of steps to deploy the image we built to Docker Hub so that you can access it anywhere.

#### 1. Create a repository on the Docker Hub

If you don’t already have an account, proceed to sign up on Docker Hub. After successfully creating an account, log in and click the Repositories tab on the navbar.""")

    st_image("repositories-docker-hub-navbar.png", 700)
    st.markdown("""
Follow the steps on the page and create a new repository named `flask-docker`.

#### 2. Log in on your local machine

The next step is to log in on your local machine to create a connection between your machine and Docker Hub.

```
docker login
```

#### 3. Rename the Docker image

When pushing an image to Docker Hub, there is a standard format that your image name has to follow. This format is specified as:

```
<your-docker-hub-username>/<repository-name>
```

Here is the command for renaming the image:

```
docker tag flask_docker <your-docker-hub-username>/flask-docker
```

#### 4. Create a repository on the Docker Hub

The final step is to push the image to Docker Hub by using the following command:

```
docker push <your-docker-hub-username>/flask-docker
```

This is what you should see upon successful deployment:""")

    st_image("successful-flask-docker-app.png", 600)

    st.markdown("""
## Deploying our app to Heroku

You should create Her

As mentioned earlier, Heroku is a platform where developers can build and run applications in the cloud. If you don't already have an account with Heroku, you can create one [here](https://signup.heroku.com/).

We'll now proceed to deploy our containerized application to Heroku with the following steps:

#### 1: Log in to Heroku

```
heroku login
```

If you’ve not previously logged in to your Docker Hub account, you’ll be required to do this to proceed.

```
docker login --username=<your-username> --password=<your-password>
```

If you get authorization errors now (or at any other point in this procedure), then running the following command might fix the problem (although you probably shouldn't have to run it):

```
heroku auth:token | docker login --username=<your-username> registry.heroku.com --password-stdin
```

#### 2: Create Heroku app

The app is the fundamental unit of Heroku. Here, you will create a Heroku app to host your Flask app.

```
heroku create <app-name>
```

#### 3: Create a Procfile

A [Procfile](https://devcenter.heroku.com/articles/procfile) contains commands that your application runs on Heroku when starting up.

Create a file and name it Procfile without an extension. Then add the following to the file:

```
web: gunicorn app:app
```

#### 4: Push the app to Heroku

```
heroku container:push web --app <app-name>
```

#### 5: Release the image

```
heroku container:release web --app <app-name>
```
""")
    st.markdown("")
    st.markdown("")

    st.info("""
Let's review all of these steps. With `flask-docker` as our app name, and `flask_docker` as our temporary name (which gets changed after the `docker tag` command), we have the following:

```
docker login
(one time only) heroku login
(one time only) heroku create <app_name>
(create Procfile containing `web: gunicorn app:app`)
docker image build -t <temporary_name> . --progress=plain
docker tag <temporary_name> <docker_username>/<app_name>
docker push <docker_username>/<app_name>
heroku container:push web --app <app_name>
heroku container:release web --app <app_name>
```""")

    st.markdown("")
    st.markdown("")
    st.markdown("""
You can now proceed to view your application on Heroku with the URL:

```
https://<app-name>.herokuapp.com/
```

## Extra reading

Congratulations, you've finished building your first app! 

If you want, you can progress immediately to the second set of exercises. Alternatively, you can read some of the mateiral below first, which should give you a better idea some Docker features and best practices:

* [Best practices from Docker documentation](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [Video: Docker best practices](https://www.youtube.com/watch?v=8vXoMqWgbQQ) (18 mins)
""")

def section_2():
    pass

def section_empty():
    st.markdown("""Coming soon!

In the meantime, the following readings might help:

* [Guide to getting started with Docker](https://docs.docker.com/get-started/) \*
* [Build and deploy a Flask app using Docker](https://blog.logrocket.com/build-deploy-flask-app-using-docker/) (the exercises will approximately follow the structure of this tutorial)
""")

func_list = [section_home, section_1, section_2]

page_list = ["🏠 Home", "1️⃣ Introduction to Docker", "2️⃣ Deploying a PyTorch model"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
