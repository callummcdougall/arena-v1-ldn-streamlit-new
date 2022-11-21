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

- Sign up for an account [here](https://lambdalabs.com/service/gpu-cloud](https://lambdalabs.com/service/gpu-cloud)
- Add an SSH key
	- Give it a name like `<Firstname><Lastname>` (we will refer to this as `<keyname>` from now on)
	- When you create it, it will automatically be downloaded
		- The file should have a `.pem` extension - this is a common container format for keys or certificates
- Install the VSCode remote-ssh extension:
	- https://hackmd.io/@tamera/mlab2-instructions
- Next step: creating an ssh file, and editing permissions
	- Windows
		- Having installed the SSH extension, Windows may have automatically created a .ssh file for you, and it will be placed in `C:\Users\<user>` by default
		- If it hasn't done this, then you should create one yourself (the windows command prompt command is `md C:\Users\<user>\.ssh`)
		- Move your downloaded SSH key into this folder
		- Set permissions:
			- Right click on file, press “Properties”, then go to the “Security” tab
			- Click “Advanced”, then “Disable inheritance” in the window that pops up
				- `instruction1.png`
			- Choose the first option “Convert inherited permissions…”
				- `instruction2.png`
			- Go back to the “Security” tab, click "Edit" to change permissions, and remove every user except the owner (you can check who the owner is by going back to "Security -> Advanced" and looking for the "Owner" field at the top of the window)
	- Linux / MacOS
		- Make your `.ssh` directory using the commands `mkdir -p ~/.ssh` then `chmod 700 ~/.ssh`
		- Set permissions on the key: `chmod 600 ~/.ssh/<keyname>`
- Go back to the LL page, go to "instances", click "Launch instance"
	- You'll see several options, some of them might be greyed out if unavailable
	- Pick a cheap one (we're only interested in testing this at the moment, and at any rate even a relatively cheap one will probably be more powerful than the one you're currently using in your laptop)
	- Enter your SSH key name
	- Choose a region (this doesn't matter)
	- Once you finish this process, you should see your GPU instance is running (`gpu_instance.png`). You should also see an SSH LOGIN field, which will look something like: `ssh ubuntu@<ip-address>`
- Set up your config file
	- Config files remove the need to use long command line arguments, e.g. `ssh -i ~/.ssh/FirstnameLastname.pem ubuntu@instance-ip-address`
	- Click on the ![[Pasted image 20221120120748.png]] button in the bottom left, choose "Open SSH Configuration File...", then click `C:\Users\<user>\.ssh\config`
	- An empty config file will open, copy in the following information (see file)
		- You should get the HostName and User from the SSHI LOGIN field in the LL table
		- You should get the SSH filename path from where you saved it
- Everyday instructions
	- Green button again, choose "Connect to Host..." then find your host
	- You'll be asked if you want to install the recommended extensions for Python - click yes
	- Click on the explorer icon, and open the directory `ubuntu`
	- Run `!nvidia-smi` in a Python or Notebook file to check your GPU is working
	- Easiest way to upload stuff - drag and drop!
	- You'll also need to choose a Python interpreter. Choose the conda or miniconda one if it's available, if not then choose the top-listed version.
		- You'll probably need to `!pip install` some libraries, e.g. plotly and einops""")

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
