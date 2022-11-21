import os
if not os.path.exists("./images"):
    os.chdir("./ch2")

from st_dependencies import *
styling()

def section_home():
    st.markdown("""
## 1️⃣ Lambda Labs

This page provides a guide for how to get set up on Lambda Labs (with different instructions depending on your OS). Once you finish this, you should be able to run large models.

There is also some reading material, which provides an overview of what GPUs are and how they work, as well as some of the topics we'll be returning to in later parts of this chapter.
""")

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#reading">Reading</a></li>
   <li><a class="contents-el" href="#introduction-lambda-labs">Introduction - Lambda Labs</a></li>
   <li><a class="contents-el" href="#instructions-for-signing-up">Instructions for signing up</a></li>
   <li><a class="contents-el" href="#vscode-remote-ssh-extension">VSCode remote-ssh extension</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#windows">Windows</a></li>
       <li><a class="contents-el" href="#linux-/-macos">Linux / MacOS</a></li>
   </ul></li>
   <li><a class="contents-el" href="#launch-your-instance">Launch your instance</a></li>
   <li><a class="contents-el" href="#set-up-your-config-file">Set up your config file</a></li>
   <li><a class="contents-el" href="#connect-to-your-instance)">Connect to your instance</a></li>
   <li><a class="contents-el" href="#exercise-use-your-gpu-to-speed-up-training-loops">Exercise - use your GPU to speed up training loops</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
## Reading

* [Techniques for Training Large Neural Networks](https://openai.com/blog/techniques-for-training-large-neural-networks/)
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)

## Introduction - Lambda Labs

Lambda Labs is a service giving you access to higher-quality GPUs than you are likely to find in your laptop. Knowing how to run models on GPUs is essential for performing large-scale experients.

In later sections of this chapter we'll look at multi-GPU setups, but for now we'll just stick to the basics: setting up a single GPU, and SSHing into it.
""")
    st.error("""Warning - **Lambda Labs charge by the hour for GPU usage**. It's currently unclear whether we'll be able to reimburse participants for their GPU usage, so we recommend not using more than you would be prepared to pay for. However, the costs should be pretty small (on the order of $5-10 per week we're using GPUs - and we won't be using them during all weeks).""")

    st.markdown(r"""
## Instructions for signing up

Sign up for an account [here](https://lambdalabs.com/service/gpu-cloud).

Add an **SSH key**. Give it a name like `<Firstname><Lastname>` (we will refer to this as `<keyname>` from now on).

When you create it, it will automatically be downloaded. The file should have a `.pem` extension - this is a common container format for keys or certificates.

## VSCode remote-ssh extension

The [**remote ssh extension**](https://code.visualstudio.com/docs/remote/ssh) is very useful for abstracting away some of the messy command-line based details of SSH. You should install this extension now.""")

    st_image("architecture-ssh.png", 600)

    st.markdown(r"""
At this point, the instructions differ between Windows and Linux/MacOS.

### Windows

Having installed the SSH extension, Windows may have automatically created a .ssh file for you, and it will be placed in `C:\Users\<user>` by default. If it hasn't done this, then you should create one yourself (you can do this from the Windows command prompt via `md C:\Users\<user>\.ssh`).

Move your downloaded SSH key into this folder. Then, set permissions on the SSH key (i.e. the `.pem` file):
		
* Right click on file, press “Properties”, then go to the “Security” tab.
* Click “Advanced”, then “Disable inheritance” in the window that pops up.""")
    st_image("instruction1.png", 500)
    st.markdown(r"""
* Choose the first option “Convert inherited permissions…”""")
    st_image("instruction2.png", 500)
    st.markdown(r"""
* Go back to the “Security” tab, click "Edit" to change permissions, and remove every user except the owner.
    * You can check who the owner is by going back to "Security -> Advanced" and looking for the "Owner" field at the top of the window).

### Linux / MacOS

* Make your `.ssh` directory using the commands `mkdir -p ~/.ssh` then `chmod 700 ~/.ssh`.
* Set permissions on the key: `chmod 600 ~/.ssh/<keyname>.pem`

## Launch your instance

Go back to the Lambda Labs page, go to "instances", and click "Launch instance".

You'll see several options, some of them might be greyed out if unavailable. Pick a cheap one (we're only interested in testing this at the moment, and at any rate even a relatively cheap one will probably be more powerful than the one you're currently using in your laptop). 

Enter your SSH key name. Choose a region (your choice here doesn't really matter for our purposes).

Once you finish this process, you should see your GPU instance is running:""")

    st_image("gpu_instance.png", 700)
    st.markdown(r"""You should also see an SSH LOGIN field, which will look something like: `ssh ubuntu@<ip-address>`.

## Set up your config file

Setting up a **config file** remove the need to use long command line arguments, e.g. `ssh -i ~/.ssh/<keyname>.pem ubuntu@instance-ip-address`.""")

    st.markdown(f"""Click on the {st_image("vscode-ssh.png", 35, return_html=True)} button in the bottom left, choose "Open SSH Configuration File...", then click <code>C:\\Users\\<user>\\.ssh\\config</code>.""", unsafe_allow_html=True)

    st.markdown(r"""
An empty config file will open. You should copy in the following instructions:

```c
Host <ip-address>
    IdentityFile C:\Users\<user>\.ssh\<keyname>.pem
    User <user>
```

where the IP address and user come from the **SSH LOGIN** field in the table, and the identity file is the path of your SSH key. For instance, the file I would use (corresponding to the table posted above) looks like:

```c
Host <ip-address>
    IdentityFile C:\Users\<user>\.ssh\<keyname>.pem
    User <user>
```

## Connect to your instance""")

    st.markdown(f"""Click the green button {st_image("vscode-ssh.png", 35, return_html=True)} again, and choose "Connect to Host...". Your IP address should appear as one of the hosts. Choose this option.""", unsafe_allow_html=True)
    st.markdown(r"""
A new VSCode window will open up. If you're asked if you want to install the recommended extensions for Python, click yes. If you're asked to choose an OS (Windows, Mac or Linux), choose Linux.

Click on the file explorer icon in the top-left, and open the directory `ubuntu` (or whichever directory you want to use as your working directory in this machine). 

And there you go - you're all set! 

To check your GPU is working, you can open a Python or Notebook file and run `!nvidia-smi`. You should see GPU information which matches the machine you chose from the Lambda Labs website, and is different from the result you get from running this command on your local machine. 

Another way to check out your GPU For instance is to run the PyTorch code `torch.cuda.get_device_name()`. For example, this is what I see after SSHing in:""")

    st_image("gpu_type.png", 600)
    st_image("gpu_type_2.png", 450)

    st.markdown("""
You can also use `torch.cuda.get_device_properties` (which takes your device as an argument).

Once you've verified this is working, you can start running code on GPUs. The easiest way to do this is just to drag and drop your files into the file explorer window on the left hand side.

You'll also need to choose a Python interpreter. Choose the conda or miniconda one if it's available, if not then choose the top-listed version. You'll probably need to `!pip install` some libraries, e.g. einops, fancy-einsum, and plotly if you're using it.

## Exercise - use your GPU to speed up training loops

You can now compare running code on your CPU and GPU. Try your Shakespeare transformer, or your GAN / VAE from last week. How much of a speedup is there? Can you relate this to what you read in [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)?

After this, you can try moving on to the next task: Pretraining BERT. To do well at this task, you'll need to use a GPU.
""")

func_list = [section_home, section_1]

page_list = ["🏠 Home", "1️⃣ Lambda Labs"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
