import os
if not os.path.exists("./images"):
    os.chdir("./ch2")

from st_dependencies import *
styling()

def section_home():
    st.markdown("""
## 1️⃣ Introduction to Docker

In the first half of today, you'll be going through a step-by-step process on how to set up a basic Docker project. This closely resembles the online tutorial from [LogRocket](https://blog.logrocket.com/build-deploy-flask-app-using-docker/).

## 2️⃣ Deploying a PyTorch model

Once you're familiar with the basics of Docker, you'll have a chance to deploy a model inside your app, e.g. your Shakespeare-trained transformer, or your GANs. There are multiple different ways we'll suggest for you to do this, depending on your background with web development, or tools such as Streamlit.""")

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction-to-docker">Introduction to Docker</a></li>
    <li><a class="contents-el" href="#getting-started">Getting started</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#docker">Docker</a></li>
        <li><a class="contents-el" href="#heroku">Heroku</a></li>
        <li><a class="contents-el" href="#misc">Misc.</a></li>
    </ul></li>
    <li><a class="contents-el" href="#creating-the-flask-app">Creating the Flask app</a></li>
    <li><a class="contents-el" href="#the-html-template">The HTML template</a></li>
    <li><a class="contents-el" href="#requirements-txt"><code>requirements.txt</code></a></li>
    <li><a class="contents-el" href="#setting-up-the-dockerfile">Setting up the Dockerfile</a></li>
    <li><a class="contents-el" href="#build-the-docker-image">Build the Docker image</a></li>
    <li><a class="contents-el" href="#run-the-container">Run the container</a></li>
    <li><a class="contents-el" href="#deploying-to-docker-hub">Deploying to Docker Hub</a></li>
    <li><a class="contents-el" href="#deploying-your-app-to-heroku">Deploying your app to Heroku</a></li>
    <li><a class="contents-el" href="#extra-reading">Extra reading</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#vscode">VSCode</a></li>
    </li></ul>
</ul>
""", unsafe_allow_html=True)
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

    st_image("docker_architecture.png", 600)

    st.markdown("""
Don't worry if this seems a little abstract right now - as we get more hands-on throughout the rest of this tutorial, this should all make a lot more sense.

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

Flask is an open-source, beginner-friendly web framework built on the Python programming language. Flask is suitable when you want to develop an application with a light codebase rapidly.

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

## `requirements.txt`

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
  ├── templates
  │   └── index.html
  ├── Dockerfile
  ├── Procfile
  ├── requirements.txt
  └── view.py
```

You can test that the application works before you proceed to containerize it. Run this command on your terminal within the root directory to perform this test:

```
python view.py
```

This command serves your Flask app. It should give you a URL that looks like `http://<your-ip-address>:5000/`, which you can use to run your app. You should see a page like this:""")

    st_image("flask_app.png", 700)
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


docker image build -t shakespeare . --progress=plain
docker tag shakespeare themcdouglas/firstapp
docker push themcdouglas/firstapp

heroku container:push web --app callum-firstapp
heroku container:release web --app callum-firstapp


Let's go over the instructions in this Dockerfile:

* **`FROM python:3.8-alpine`**
    * Since Docker allows us to inherit existing images, we install a Python image and install it in our Docker image.
    * `Alpine` is a lightweight Linux distro that will serve as the OS on which we install our image.
    * See [this page](https://pythonspeed.com/articles/base-image-python-docker-images/) for a discussion of which Docker base images to use for your Python app.
        * In particular, note that Alpine won't enable to use NumPy or PyTorch, which will be pretty important for our use-cases! We will discuss alternatives in the next section.
* **`COPY ./requirements.txt /app/requirements.txt`**
    * Here, we copy the requirements file and its content (the generated packages and dependencies) into the app folder of the image
    * Remember, the second path here (the one containing `app`) refers to your container - it has no relation to the `flask_docker` directory on the machine you're currently working on.
* **`WORKDIR /app`**
    * We proceed to set the working directory as `/app`, which will be the root directory of our application in the container.
    * Any subsequent commands (e.g. `COPY`, `ENTRYPOINT`, `CMD`) will refer to this directory.
* **`RUN pip install -r requirements.txt`**
    * This command installs all the dependencies defined in the `requirements.txt file` into our application within the container.
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

The flag `-t` stands for tag. This means we can now refer to this image using the tag `flask_docker`.

You can also add the flag `--progress=plain` to the end of this command, if you want to see logs during the building process.

## Run the container

After successfully building the image, the next step is to run an instance of the image. Here is how to perform this:

```
docker run -p 5000:5000 -d flask_docker
```

This command runs the container and its embedded application, each on port 5000 using a port-binding approach. The first 5000 is the port that we allocate to the container on our machine. The second 5000 is the port where the application will run on the container. Remember that our docker container has its own filesystem and ports, distinct from the filesystem and ports on our client machine.

Once you run this command, you should see the same webpage as earlier when you send the request `localhost:5000` in your browser. Also, if you open your Docker Desktop app, and select **Containers** from the menu on the left, you should see your container listed there, along with its status, ports and how long ago it started.""")

    st_image("containers.png", 700)
    st.markdown("""

## Deploying to Docker Hub

As we mentioned above, Docker Hub is a registry where Docker users can create, test, and manage containers. If you've worked with GitHub, this section will be very familiar to you.

Follow the next sequence of steps to deploy the image we built to Docker Hub so that you can access it anywhere.

#### 1. Create a repository on the Docker Hub

If you don't already have an account, proceed to [sign up](https://hub.docker.com/signup) on Docker Hub. After successfully creating an account, log in and click the Repositories tab on the navbar.""")

    st_image("repositories-docker-hub-navbar.png", 700)
    st.markdown("""
Follow the steps on the page and create a new repository named `flask-docker`.

#### 2. Log in on your local machine

The next step is to log in on your local machine to create a connection between your machine and Docker Hub.

```
docker login
```""")
    with st.expander("Help - I get 'error during connect: This error may indicate that the docker daemon is not running'."):
        st.markdown("""
This might be because you've disabled the features which cause Docker to run automatically on startup. Running the Docker Desktop app should fix this.""")

    st.markdown("""
#### 3. Rename the Docker image

When pushing an image to Docker Hub, there is a standard format that your image name has to follow. This format is specified as:

```
<your-docker-hub-username>/<repository-name>
```

Here is the command you should use to rename the image:

```
docker tag flask_docker <your-docker-hub-username>/flask-docker
```

When you do this, you should see a second image appear in the **Images** tab of your Docker Desktop app, with the new name.

#### 4. Create a repository on the Docker Hub

The final step is to push the image to Docker Hub by using the following command:

```
docker push <your-docker-hub-username>/flask-docker
```

This is what you should see upon successful deployment:""")

    st_image("successful-flask-docker-app.png", 600)

    st.markdown("""
## Deploying your app to Heroku

As mentioned earlier, Heroku is a platform where developers can build and run applications in the cloud. If you don't already have an account with Heroku, you can create one [here](https://signup.heroku.com/).

You should now proceed to deploy our containerized application to Heroku, with the following steps:

#### 1: Log in to Heroku

```
heroku login
```

If you've not previously logged in to your Docker Hub account, you'll be required to do this to proceed.

```
docker login --username=<your-username> --password=<your-password>
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
```""")

    with st.expander("Help - I get 'unauthorized: authentication required'."):
        st.markdown("""
Try running the following command (with your username):

```
heroku auth:token | docker login --username=<your-username> registry.heroku.com --password-stdin
```

and then trying again to push. If this also fails, it might be time for some Google / StackOverflow debugging!
""")

    st.markdown("""
#### 5: Release the image

```
heroku container:release web --app <app-name>
```
""")
    st.markdown("")
    st.markdown("")

    st.info("""
Let's review all of these steps. We have:

```
docker login
docker image build -t <tag-name> . --progress=plain
(for running locally) docker run -p 5000:5000 -d <tag-name>
docker tag <tag-name> <docker-username>/<repository-name>
docker push <docker-username>/<repository-name>

heroku login
heroku create <app-name>
(create Procfile containing `web: gunicorn app:app`)
heroku container:push web --app <app-name>
heroku container:release web --app <app-name>
```

where in this case, we used `flask-docker` as our tag name, and `flask_docker` as our repository name. 

When you make changes to the app, you will only need to run three commands again: the `docker push`, and the last two Heroku commands: `push` and `release`. If your changes include new dependencies (e.g. changes to `requirements.txt`), then you'll need to build a new image.

Our directory structure at this point looks like:

```
flask_docker
  ├── templates
  │   └── index.html
  ├── Dockerfile
  ├── Procfile
  ├── requirements.txt
  └── view.py
```
""")

    st.markdown("")
    st.markdown("")
    st.markdown("""
You can now proceed to view your application on Heroku with the URL:

```
https://<app-name>.herokuapp.com/
```

Alternatively, you can run the command:

```
heroku open --app <app-name>
```

and this URL should automatically open for you.

## Extra reading

Congratulations, you've finished building your first app! 

If you want, you can progress immediately to the second set of exercises. Alternatively, you can read some of the mateiral below first, which should give you a better idea some Docker features and best practices:

* [Best practices from Docker documentation](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [Video: Docker best practices](https://www.youtube.com/watch?v=8vXoMqWgbQQ) (18 mins)

### VSCode

VSCode has a very useful Docker extension. Much like its GitHub extension, this abstracts away many of the messy details of running commands in the CLI. You can do things like:

* See a list of currently active containers and images.
* View the filestructure of your container (this is especially useful to check that e.g. commands like `COPY` in your Dockerfile have worked as you expected them to).
* Run commands like pushes and pulls with docker.

However, it's recommended not to use this until you're a bit more comfortable with the CLI-based workflow.
""")

def section_2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#methods-of-deployment">Methods of deployment</a></li>
    <li><a class="contents-el" href="#streamlit">Streamlit</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#getting-started-with-streamlit">Getting started with Streamlit</a></li>
        <li><a class="contents-el" href="#deploy-streamlit-using-docker">Deploy Streamlit using Docker</a></li>
    </ul></li>
    <li><a class="contents-el" href="#flask-html-css-javascript">Flask (& HTML, CSS JavaScript)</a></li>
    <li><a class="contents-el" href="#conclusion">Conclusion</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
Now that you've made a basic Docker app, it's time to deploy one of the models you trained earlier in the course!

This can be whichever one you'd like - you can use your MNIST from the first week, or you can use more advanced models like your Shakespeare transformer, or your GAN / VAE from the contrastive modelling section.

## Methods of deployment

Now, we come to ways you can deploy your app.

One point to address first - in the previous tutorial we used the `python:3.8-alpine` image. This suffices for basic applications, but it won't now that we're trying to import and use packages like NumPy and PyTorch. Messing around with the right images and `RUN` commands can be a pretty big pain! 

I found that using the following the following in my Dockerfile worked:

```docker
FROM python:3.8-slim

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r requirements.txt
```

The `install torch` line was taken from [this site](https://pytorch.org/get-started/locally/), using the following settings: Stable, Linux, Pip, Python, CPU. Note that using CPU rather than CUDA results in a smaller image, but also means that this model can't be run on a GPU. Later in this chapter we'll try and build models that can be run on GPUs (we'll use **FastAPI** to do this rather than Flask), but for now most of the models we've built will be able to perform inference on a CPU without too much stress.

Building this image will take some time since, PyTorch download and installation takes a while.

One gotcha here - if your app actually contains PyTorch models or state dictionaries which you load during runtime, make sure these are stored on the CPU. In the case of a state dictionary, you can move it to the CPU using code like:

```python
state_dict_cpu = {k: v.cpu() for k, v in state_dict_cuda.items()}
t.save(state_dict_cpu, "state_dict.pt")
```

## Streamlit

Streamlit is the framework these exercises are being hosted with (i.e. these pages were designed using Streamlit). But in fact, Streamlit can be used for a lot more than serving markdown files with navbars! It was **specifically designed for developing and deploying interactive data science dashboards and machine learning models.**

One major advantage of Streamlit is that the entire thing is written in Python, as opposed to the suggested task above which requires you to have familiarity with HTML and JavaScript at least. This makes it extremely easy to pick up and become proficient in, even if you have no experience of web development. So if the **HTML, CSS & JavaScript** option below seems intimidating, I'd recommend trying this one.

### Getting started with Streamlit

Gettting started is as simple as installing Streamlit with `pip install streamlit`, then running `streamlit hello`. A Streamlit page will open that demos some of its main features, and the page also provides links to other help pages and documentation.

### Deploy Streamlit using Docker

Luckily, Streamlit has consistently good documentation, including a page on how to [deploy Streamlit using Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker). Most of the stuff on this page should be familiar to you (or at least understandable) given the context of the section 1️⃣ Introduction to Docker.


## Flask (& HTML, CSS JavaScript)

For people with any front-end web development background, this might be a fun project. Even if you don't, this could be a good opportunity to tip your toe in! I'd recommend the [CS50x course](https://cs50.harvard.edu/x/2022/notes/8/) - specifically week 8 - for a speed-run introduction to HTML, CSS and JavaScript and how they work together. If you're looking for a longer intro, then you might want to check out [CS50's Web Programming with Python and Javascript](https://cs50.harvard.edu/web/2020/). However, it's worth emphasising that to fully treat these topics would take us quite far outside the bounds of this course material!

Once you understand the basics, you should be able to create simple apps like this:""")

    st_image("vanilla-shakespeare.png", 600)

    st.markdown(r"""
This was created using the following basic method:

### 1. Adding the appropriate Python files to the directory

Below is what the directory looked like at the end (note that this represents just one possible design choice, and it's possible this isn't in line with best practices!).

```
flask_docker
  ├── my_transformer
  │   ├── 100-0.txt                  
  │   ├── load_shakespeare.py
  │   ├── model_state_dict.pt
  │   ├── sampling_methods.py
  │   └── transformer_architecture.py
  ├── templates
  │   └── index.html
  ├── Dockerfile
  ├── Procfile
  ├── requirements.txt
  └── view.py
```

### 2. Adding [JavaScript Forms](https://www.w3schools.com/js/js_validation.asp) to `index.html`

Here is an example of how you can use a form, which just takes initial text and temperature as inputs, and also has a submit button:

```js
<form action="{{ url_for('submit') }}" method="post">
    <div id="text-input">
        <textarea name="text" rows="4" cols="50" placeholder="Enter your text here, and press submit."></textarea>
    </div>

    <div>
        <label for="temperature">Temperature:</label>
        <input type="number" name="temperature" min="0.0" max="10.0" value="1.0" step="0.05">
    </div>

    <div>
        <input type="submit" value="Generate text!">
    </div>
</form>
```

See [this page](https://www.w3schools.com/html/html_form_input_types.asp) to view and test out the other types of `<input>` elements you can add to forms.

Note the `action="{{ url_for('submit') }}` argument inside the form definition. This tells Flask where the data is sent to on submit (see the following section for how our `submit` page handles this data). The `method` argument tells Flask how to submit the data, either as a query string (GET) or form data (POST). Here, I'm using POST so that the values aren't visible in the URL. You could use GET in this case if you wanted, although it would be somewhat less natural.

### 3. Replace the `home` function (decorated with `@app.route('/')`) with:

```python
@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():

    # Get information from the `request.form` object, using elements' `name` attribute
    temperature = float(request.form["temperature"])
    initial_text = request.form["text"]

    # Generate model output (not shown)

    # Return `index.html`, with model output passed as an argument
    return render_template('index.html', model_output=model_output)
```

### 4. Adding a location for the model output to `index.html`

```html
<pre>{{model_output}}</pre>
```

See [this section of the Flask introduction](https://www.geeksforgeeks.org/python-introduction-to-web-development-using-flask/#:~:text=Sending%20Form%20Data%20to%20the%20HTML%20File%20of%20Server%3A) to see how the double curly brackets syntax can be used to send information that is rendered in your html pages.

---

## Conclusion

Once you've built an app, please share it with the group! It would be great to assemble a gallery of Docker apps that people on this programme have created.

Later in this chapter, we'll use Lambda Labs to pre-train our BERT model on some powerful GPUs, then bring everything full-circle by deploying our apps.
""")

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