import streamlit as st
import base64
import platform
import os
if not os.path.exists("./images"):
    os.chdir("./prereqs")

is_local = (platform.processor() != "")

def st_image(name, width):
    with open("images/" + name, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    img_html = f"<img style='width:{width}px;max-width:100%;margin-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
    st.markdown(img_html, unsafe_allow_html=True)

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

st_image("main.png", 320)
st.markdown(r"""
# Prerequisites for ARENA

This is a document containing recommended material to study, in advance of the ARENA programme. None of it is compulsory and some resources are likely to be much more helpful than others, and **we don't expect you to study everything on this list** given the programme is starting so soon. We denote very high and high-priority resources with a double and single asterisk respectively, so if you have limited time then prioritise these. It is **strongly recommended** to at least read over everything with a double asterisk. You can move the slider below to show all material, just single/double-asterisked material, or just double-asterisked material respectively.

Also, you should try and prioritise areas you think you might be weaker in than others (for instance, if you have a strong SWE background but less maths experience then you might want to spend more time on the maths sections). You can also return to this document throughout the programme, if there are any areas you want to brush up on.

The content is partially inspired by a similar doc handed out by Redwood to participants before the start of MLAB, as well as by pre-prerequisite material provided by Jacob Hilton on his GitHub page.

Throughout this document, there are some questions peppered in, which will be indicated by red boxes. You should be able to answer all of these questions, once you've read through the relevant material.

If you are reading this, and know of any good material for these topics which we've missed out, please let us know and we might be able to add it in!""")

cols = st.columns([1, 4, 1])
with cols[1]:
    slider = st.slider(label="Move the slider to change the amount of material shown.", min_value=0, max_value=2, value=0) #, help="test")

def show_all():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#maths">Maths</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#neural-networks**">Neural Networks**</a></li>
        <li><a class="contents-el" href="#linear-algebra**">Linear algebra**</a></li>
        <li><a class="contents-el" href="#probability**">Probability**</a></li>
        <li><a class="contents-el" href="#calculus**">Calculus**</a></li>
        <li><a class="contents-el" href="#statistics*">Statistics*</a></li>
        <li><a class="contents-el" href="#information-theory*">Information theory*</a></li>
    </ul></li>
    <li><a class="contents-el" href="#programming">Programming</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#python">Python**</a></li>
        <li><a class="contents-el" href="#libraries">Libraries</a></li>
        <li><ul class="contents">
            <li><a class="contents-el" href="#numpy">numpy**</a></li>
            <li><a class="contents-el" href="#pytorch">pytorch**</a></li>
            <li><a class="contents-el" href="#einops-and-einsum">einops and einsum*</a></li>
            <li><a class="contents-el" href="#typing">typing*</a></li>
            <li><a class="contents-el" href="#plotly">plotly</a></li>
            <li><a class="contents-el" href="#ipywidgets">ipywidgets</a></li>
            <li><a class="contents-el" href="#streamlit">streamlit</a></li>
        </ul></li>
    </ul></li>
    <li><a class="contents-el" href="#other-programming-concepts-tools">Other programming concepts / tools</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#basic-coding-skills">Basic coding skills**</a></li>
        <li><a class="contents-el" href="#vscode">VSCode**</a></li>
        <li><a class="contents-el" href="#git">Git**</a></li>
        <li><a class="contents-el" href="#jupyter-notebook-colab">Jupyter Notebook / Colab*</a></li>
        <li><a class="contents-el" href="#unix">Unix*</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
## Maths

### Neural Networks**

We won't assume any deep knowledge of neural networks or machine learning before the programme starts, but it's useful to have an idea of the basis so that the first week doesn't have quite as steep a learning curve. The best introductory resources here are 3B1B's videos on neural networks:

- **[But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk)**
- **[Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w)**
- **[What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U)**

You should prioritise the first two videos in this sequence.

Some questions you should be able to answer after this:

""")
    st.error(r"""
- **Why do you need activation functions? Why couldn't you just create a neural network by connecting up a bunch of linear layers?**
- **What makes neural networks more powerful than basic statistical methods like linear regression?**
- **What are the advantages of ReLU activations over sigmoids?**""")

    st.markdown(r"""

### Linear algebra**

Linear algebra lies at the core of a lot of machine learning. Insert obligatory xkcd:

""")

    st_image("xkcd_1.png", 300)

    st.markdown(r"""

Here is a list of things you should probably be comfortable with:

- Linear transformations - what they are, and why they are important
    - See [this video](https://www.youtube.com/watch?v=kYB8IZa5AuE) from 3B1B""")

    st.error("""
**What is the problem in trying to create a neural network using only linear transformations?**""")

    st.markdown(r"""
- How [matrix multiplication works](http://mlwiki.org/index.php/Matrix-Matrix_Multiplication)
- Basic matrix properties: rank, trace, determinant, transpose
- Bases, and basis transformations""")

    st.error(r"""
**Read the “[Privileged vs Free Basis](https://transformer-circuits.pub/2021/framework/index.html#def-privileged-basis)” section of Anthropic's “Mathematical Framework for Transformer Circuits” paper. Which kinds of operations might lead to the creation of privileged bases? Which linear operations would preserve a privileged basis? (you might want to come back to this question once you're more familiar with the basics of neural networks and activation functions)**""")
    st.markdown(r"""
- Eigenvalues and eigenvectors
- Different types of matrix, and their significance (e.g. symmetric, orthogonal, identity, rotation matrices)

[This video series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) by 3B1B provides a good overview of these core topics (although you can probably skip it if you already have a reasonably strong mathematical background).

If you have a lot more time, [Linear Algebra Done Right](https://link.springer.com/book/10.1007/978-3-319-11080-6) is the canonical textbook for covering this topic (although it will probably cover much more than you need to know). Alternatively, Neel Nanda has two [YouTube](https://www.youtube.com/watch?v=GkPhwnvRe-8) [videos](https://www.youtube.com/watch?v=0EB23unfLSU) covering linear algebra extensively. 

### Probability**

It's essential to understand the rules of probability, expected value and standard deviation, and helpful to understand independence and the normal distribution.

""")

    st.error(r"""**What is the expected value and variance of the sum of two normally distributed random variables $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_1 \sim N(\mu_2, \sigma_2^2)$? Can you prove this mathematically?**""")

    st.markdown(r"""

### Calculus**

It's essential to understand differentiation and partial differentiation, and helpful to understand the basics of vector calculus including the chain rule and Taylor series.

Again, 3Blue1Brown has a good [video series](https://www.3blue1brown.com/topics/calculus) on this.""")

    st.error(r"""
**Read [this PyTorch documentation page](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) on the MSELoss. What is the derivative of this loss wrt the element $y_n$ (assuming we use mean reduction)? Same question for [this page](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) on BCELoss.**""")

    st.markdown(r"""

### Statistics*

It's helpful to understand estimators and standard errors.

[This page](https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/07%3A_Point_Estimation/7.01%3A_Estimators) contains a pretty useful overview of several concepts related to statistical estimators (although not all of it is essential).

Additionally, the first 1 hr 15 mins of [Neel Nanda's YouTube video on linear algebra](https://www.youtube.com/watch?v=GkPhwnvRe-8) also talks about its relation to statistics; covering topics like linear regression and hypothesis testing.

### Information theory*

It's helpful to understand information, entropy and KL divergence. These play key roles in interpreting loss functions, and will be especially important in Week 5 (modelling objectives).

[Elements of Information Theory](http://staff.ustc.edu.cn/~cgong821/Wiley.Interscience.Elements.of.Information.Theory.Jul.2006.eBook-DDU.pdf) by Thomas M. Cover is the textbook recommended by Jacob Hilton. It will probably cover more than you need to know, and this need not be prioritised over other items on the list.

For an overview of Kullback Leibler divergence (an important concept in information theory and machine learning), see [Six (and a half) intuitions for KL divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence). Note that this probably won't make sense if you don't already have a solid grasp of what entropy is.""")

    st.error(r"""
**Why is cross entropy loss a natural choice when training classifiers? The post above might help answer this question (note that KL divergence and cross entropy differ by a constant).**""")
    st.markdown(r"""

## Programming

### Python

It's important to be strong in Python, because this is the language we'll be using during the programme. As a rough indication, we expect you to be comfortable with at least 80-90% of the material  [here](https://book.pythontips.com/en/latest/), up to 21. for/else. For a more thorough treatment of Python's core functionality, see [here](https://docs.python.org/3/tutorial/).

### Libraries

The following libraries would be useful to know to at least a basic level, before the course starts.

### [`numpy`](https://numpy.org/)**

Being familiar with NumPy is a staple for working with high-performance Python. Additionally, the syntax for working with NumPy arrays is very similar to how you work with PyTorch tensors (often there are only minor differences, e.g. Torch tends to use the keyword axis where NumPy uses dim). Working through [these 100 basic NumPy exercises](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.ipynb) would be a good idea, or if you're comfortable with NumPy already then you could try doing them in PyTorch (see below).

### [`pytorch`](https://pytorch.org/)**

We will be starting week 0 of the programme with some structured exercises designed to get everyone familiar with working in PyTorch. However, the more comfortable you are with PyTorch going in, the easier you'll probably find this. PyTorch has several useful tutorials, and to get comfortable working with tensors you might want to implement the 100 basic NumPy exercises linked to above, using PyTorch instead.

### [`einops`](https://einops.rocks/1-einops-basics/) and [`einsum`](https://pypi.org/project/fancy-einsum/)*

These are great libraries to get comfortable with, when manipulating tensors. If you're comfortable using them, then you can say goodbye to awkward NumPy/PyTorch methods like `transpose`, `permute` and `squeeze`! We'll have a few einops and einsum exercises on W0D2, but the more comfortable you are with these libraries the faster you'll be.

For einops, you can read through the examples up to “Fancy examples in random order”. It's worth trying to play around with these in your own Jupyter notebook, to get more comfortable with them. 

For einsum, [this page](https://rockt.github.io/2018/04/30/einsum) provides a basic intro to einstein summation convention, and shows some example tensor implementations. Note that `fancy_einsum.einsum` allows you to use words to represent dimensions, in the same way as `einops` does, and for that reason it's usually preferred (since it's more explicit, and easier to troubleshoot when the code isn't working). 

### [`typing`](https://docs.python.org/3/library/typing.html)*

Type-checking Python functions that you write is a great way to catch bugs, and keep your code clear and readable. Python isn't a strongly-typed language so you won't get errors from using incorrect type specifications unless you use a library like MyPy. However, if you're using VSCode then you can pair this library with a really useful automatic type checker (see next section).

### [`plotly`](https://plotly.com/python/)

Plotly is an interactive graphing library which is great for presenting results and investigating data. If you're already very familiar with a different Python plotting library (e.g. matplotlib) then I wouldn't recommend re-learning Plotly, but if you aren't already very familiar with matplotlib or you're open to learning Plotly, I'd strongly recommend giving it a try!

[Here](https://github.com/callummcdougall/plotly-widgets/blob/main/WidgetsPlotlyGuide.ipynb) is a notebook I (Callum) wrote to provide an introduction to Plotly's basic features. I intended it as a summation of all useful material from the Plotly documentation pages, so it's pretty long (although hopefully still much shorter than reading through all of Plotly's documentation). It also discusses Jupyter Widgets (see below) which might be less relevant in the context of this course. You can also find the same data [here](https://callummcdougall-plotly-widgets--home-ujpce1.streamlitapp.com/).

### [`ipywidgets`](https://ipywidgets.readthedocs.io/en/stable/index.html)

Widgets work very well with Plotly in Jupyter Notebooks; you can see some examples in the [notebook above](https://github.com/callummcdougall/plotly-widgets/blob/main/WidgetsPlotlyGuide.ipynb), or at [this Plotly documentation page](https://plotly.com/python/figurewidget-app/). This can be a fun way to create interactive visualisations with sliders and dropdown menus etc.

Learning this library shouldn't be prioritised over other things in this document.

### [`streamlit`](https://share.streamlit.io/)

Streamlit is a cool library for building and sharing data-based applications. It integrates very nicely with Plotly (see above), can be hosted on your personal GitHub, and is very intuitive & easy to learn relative to other libraries with similar features (e.g. Dash). This is not compulsory, but if you like the look of Streamlit then you might want to think about using it as a way to submit (or even make public) your end-of-week or capstone projects. See [this simple page](https://callummcdougall-new-repo-home-qghfdp.streamlitapp.com/) I (Callum) made for visualising results from Neel Nanda's grokking paper, as an example of what Streamlit & Plotly can do.

## Other programming concepts / tools

### Basic coding skills**

If you've been accepted into this programme, then you probably already have this box ticked! However, polishing this area can't hurt. LeetCode is a good place to keep basic coding skills sharp, in particular practising the planning and implementation of functions in the medium-hard sections of LeetCode might be helpful. Practising problems on [Project Euler](https://projecteuler.net/) is another idea.

### VSCode**

Although many of you might already be familiar with Jupyter Notebooks, we recommend working through structured exercises using VSCode. This is a powerful text editor which provides more features than Jupyter Notebooks. Some features it has are:

- **Shortcuts**
    
    These are much more powerful than anything offered by Jupyter Notebooks. Here are a list of particularly useful ones. You can see how they all work in more detail [here](https://www.geeksforgeeks.org/visual-studio-code-shortcuts-for-windows-and-mac/).
    
    """)

    st_image("vscode-shortcuts.png", 750)

    st.markdown(r"""
    
- **Type checking**
    
    We discussed the `typing` module in a section above. This is particularly powerful when used alongside VSCode's type checker extension. You can activate typing by going to the `settings.json` file in VSCode, and adding this line:
    
    ```json
    {
        "python.analysis.typeCheckingMode": "basic"
    }
    ```
    
    You can open the `settings.json` file by first opening VSCode's Command Palette (see the shortcuts above), then finding the option **Preferences: Open User Settings (JSON)**.
    
    If you're finding that the type checker is throwing up too many warnings, you can suppress them by adding the comment `# type: ignore` at the end of a line, or using the `cast` function from the `typing` library to let the Python interpreter know what type it's dealing with. Sometimes, this is the best option. However, in general you should try and avoid doing this when you can, because type checker warnings usually mean there's a better way for you to be writing your code.
    
- **Notebook functionality**
    
    Although VSCode does provide an extension which acts just like a Jupyter Notebook, it actually has a much more useful feature. Python files can also be made to act like notebooks, by adding the line`#%%` which act as cell dividers. In this way, you can separate chunks of code and run them individually (and see their output in a new window). See [this page](https://code.visualstudio.com/docs/python/jupyter-support-py) for a further explanation.
    
- **Debugger**
    
    The VSCode debugger is a great tool, and can be much more powerful and efficient than the standard practice of adding lines to print information about your output! You can set breakpoints in your code, and closely examine the local variables which are stored at that point in the program. More information can be found on [this page](https://lightrun.com/debugging/debug-python-in-vscode/).

### Git**

Git is a piece of version control software, designed for tracking and managing changes in a set of files. It can be very powerful, but also a bit of a headache to use. Insert obligatory second xkcd:

""")

    st_image("xkcd_2.png", 300)

    st.markdown(r"""

If you already have a strong SWE background then you might not need to spend as much time on this section. Otherwise, here are a few resources you might find useful for learning the basics of Git:

- [An Intro to Git and GitHub for Beginners](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners)
- [Git Immersion](https://gitimmersion.com/index.html)
- [Git cheat sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)

Ideally, you should feel comfortable with the following:

- Cloning a repository
- Creating and switching between branches
- Staging and committing changes
- Pushing branches

### Jupyter Notebook / Colab*

Jupyter Notebooks still have some advantages over VSCode, primarily in data exploration and visualisation. At the end of each week you will be uploading your projects to GitHub, and doing this in notebook form might be a good idea.

Colab has a similar structure to Jupyter Notebooks, but it provides several additional features, most importantly GPU access. It's not yet confirmed how much we'll be using Colab and Notebooks relative to VSCode, so this section is pending updates, but it might well be worth playing around with a Colab notebook and getting the hang of how it works, e.g. how to use the built-in keyboard shortcuts to create/delete/split/run cells.

### Unix*

This won't matter much for the first 2 weeks, but it may come into play more during the “training at scale” week. [**Learning the Shell**](http://www.ee.surrey.ac.uk/Teaching/Unix/) provides a comprehensive introduction, as does **[UNIX Tutorial for Beginners](http://www.ee.surrey.ac.uk/Teaching/Unix/).**
""")

def show_single_or_double():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#maths">Maths</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#neural-networks**">Neural Networks**</a></li>
        <li><a class="contents-el" href="#linear-algebra**">Linear algebra**</a></li>
        <li><a class="contents-el" href="#probability**">Probability**</a></li>
        <li><a class="contents-el" href="#calculus**">Calculus**</a></li>
        <li><a class="contents-el" href="#statistics*">Statistics*</a></li>
        <li><a class="contents-el" href="#information-theory*">Information theory*</a></li>
    </ul></li>
    <li><a class="contents-el" href="#programming">Programming</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#python">Python**</a></li>
        <li><a class="contents-el" href="#libraries">Libraries</a></li>
        <li><ul class="contents">
            <li><a class="contents-el" href="#numpy">numpy**</a></li>
            <li><a class="contents-el" href="#pytorch">pytorch**</a></li>
            <li><a class="contents-el" href="#einops-and-einsum">einops and einsum*</a></li>
            <li><a class="contents-el" href="#typing">typing*</a></li>
        </ul></li>
    </ul></li>
    <li><a class="contents-el" href="#other-programming-concepts-tools">Other programming concepts / tools</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#basic-coding-skills">Basic coding skills**</a></li>
        <li><a class="contents-el" href="#vscode">VSCode**</a></li>
        <li><a class="contents-el" href="#git">Git**</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""

## Maths

### Neural Networks**

We won't assume any deep knowledge of neural networks or machine learning before the programme starts, but it's useful to have an idea of the basis so that the first week doesn't have quite as steep a learning curve. The best introductory resources here are 3B1B's videos on neural networks:

- **[But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk)**
- **[Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w)**
- **[What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U)**

You should prioritise the first two videos in this sequence.

Some questions you should be able to answer after this:

""")
    st.error(r"""
- **Why do you need activation functions? Why couldn't you just create a neural network by connecting up a bunch of linear layers?**
- **What makes neural networks more powerful than basic statistical methods like linear regression?**
- **What are the advantages of ReLU activations over sigmoids?**""")

    st.markdown(r"""

### Linear algebra**

Linear algebra lies at the core of a lot of machine learning. Insert obligatory xkcd:

""")

    st_image("xkcd_1.png", 300)

    st.markdown(r"""

Here is a list of things you should probably be comfortable with:

- Linear transformations - what they are, and why they are important
    - See [this video](https://www.youtube.com/watch?v=kYB8IZa5AuE) from 3B1B""")

    st.error("""
**What is the problem in trying to create a neural network using only linear transformations?**""")

    st.markdown(r"""
- How [matrix multiplication works](http://mlwiki.org/index.php/Matrix-Matrix_Multiplication)
- Basic matrix properties: rank, trace, determinant, transpose
- Bases, and basis transformations""")

    st.error(r"""
**Read the “[Privileged vs Free Basis](https://transformer-circuits.pub/2021/framework/index.html#def-privileged-basis)” section of Anthropic's “Mathematical Framework for Transformer Circuits” paper. Which kinds of operations might lead to the creation of privileged bases? Which linear operations would preserve a privileged basis? (you might want to come back to this question once you're more familiar with the basics of neural networks and activation functions)**""")
    st.markdown(r"""
- Eigenvalues and eigenvectors
- Different types of matrix, and their significance (e.g. symmetric, orthogonal, identity, rotation matrices)

[This video series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) by 3B1B provides a good overview of these core topics (although you can probably skip it if you already have a reasonably strong mathematical background).

If you have a lot more time, [Linear Algebra Done Right](https://link.springer.com/book/10.1007/978-3-319-11080-6) is the canonical textbook for covering this topic (although it will probably cover much more than you need to know). Alternatively, Neel Nanda has two [YouTube](https://www.youtube.com/watch?v=GkPhwnvRe-8) [videos](https://www.youtube.com/watch?v=0EB23unfLSU) covering linear algebra extensively. 

### Probability**

It's essential to understand the rules of probability, expected value and standard deviation, and helpful to understand independence and the normal distribution.

""")

    st.error(r"""**What is the expected value and variance of the sum of two normally distributed random variables $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_1 \sim N(\mu_2, \sigma_2^2)$? Can you prove this mathematically?**""")

    st.markdown(r"""

### Calculus**

It's essential to understand differentiation and partial differentiation, and helpful to understand the basics of vector calculus including the chain rule and Taylor series.

Again, 3Blue1Brown has a good [video series](https://www.3blue1brown.com/topics/calculus) on this.""")

    st.error(r"""
**Read [this PyTorch documentation page](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) on the MSELoss. What is the derivative of this loss wrt the element $y_n$ (assuming we use mean reduction)? Same question for [this page](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) on BCELoss.**""")

    st.markdown(r"""

### Statistics*

It's helpful to understand estimators and standard errors.

[This page](https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/07%3A_Point_Estimation/7.01%3A_Estimators) contains a pretty useful overview of several concepts related to statistical estimators (although not all of it is essential).

Additionally, the first 1 hr 15 mins of [Neel Nanda's YouTube video on linear algebra](https://www.youtube.com/watch?v=GkPhwnvRe-8) also talks about its relation to statistics; covering topics like linear regression and hypothesis testing.

### Information theory*

It's helpful to understand information, entropy and KL divergence. These play key roles in interpreting loss functions, and will be especially important in Week 5 (modelling objectives).

[Elements of Information Theory](http://staff.ustc.edu.cn/~cgong821/Wiley.Interscience.Elements.of.Information.Theory.Jul.2006.eBook-DDU.pdf) by Thomas M. Cover is the textbook recommended by Jacob Hilton. It will probably cover more than you need to know, and this need not be prioritised over other items on the list.

For an overview of Kullback Leibler divergence (an important concept in information theory and machine learning), see [Six (and a half) intuitions for KL divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence). Note that this probably won't make sense if you don't already have a solid grasp of what entropy is.""")

    st.error(r"""
**Why is cross entropy loss a natural choice when training classifiers? The post above might help answer this question (note that KL divergence and cross entropy differ by a constant).**""")
    st.markdown(r"""

## Programming

### Python

It's important to be strong in Python, because this is the language we'll be using during the programme. As a rough indication, we expect you to be comfortable with at least 80-90% of the material  [here](https://book.pythontips.com/en/latest/), up to 21. for/else. For a more thorough treatment of Python's core functionality, see [here](https://docs.python.org/3/tutorial/).

### Libraries

The following libraries would be useful to know to at least a basic level, before the course starts.

### [`numpy`](https://numpy.org/)**

Being familiar with NumPy is a staple for working with high-performance Python. Additionally, the syntax for working with NumPy arrays is very similar to how you work with PyTorch tensors (often there are only minor differences, e.g. Torch tends to use the keyword axis where NumPy uses dim). Working through [these 100 basic NumPy exercises](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.ipynb) would be a good idea, or if you're comfortable with NumPy already then you could try doing them in PyTorch (see below).

### [`pytorch`](https://pytorch.org/)**

We will be starting week 0 of the programme with some structured exercises designed to get everyone familiar with working in PyTorch. However, the more comfortable you are with PyTorch going in, the easier you'll probably find this. PyTorch has several useful tutorials, and to get comfortable working with tensors you might want to implement the 100 basic NumPy exercises linked to above, using PyTorch instead.

### [`einops`](https://einops.rocks/1-einops-basics/) and [`einsum`](https://pypi.org/project/fancy-einsum/)*

These are great libraries to get comfortable with, when manipulating tensors. If you're comfortable using them, then you can say goodbye to awkward NumPy/PyTorch methods like `transpose`, `permute` and `squeeze`! We'll have a few einops and einsum exercises on W0D2, but the more comfortable you are with these libraries the faster you'll be.

For einops, you can read through the examples up to “Fancy examples in random order”. It's worth trying to play around with these in your own Jupyter notebook, to get more comfortable with them. 

For einsum, [this page](https://rockt.github.io/2018/04/30/einsum) provides a basic intro to einstein summation convention, and shows some example tensor implementations. Note that `fancy_einsum.einsum` allows you to use words to represent dimensions, in the same way as `einops` does, and for that reason it's usually preferred (since it's more explicit, and easier to troubleshoot when the code isn't working). 

### [`typing`](https://docs.python.org/3/library/typing.html)*

Type-checking Python functions that you write is a great way to catch bugs, and keep your code clear and readable. Python isn't a strongly-typed language so you won't get errors from using incorrect type specifications unless you use a library like MyPy. However, if you're using VSCode then you can pair this library with a really useful automatic type checker (see next section).

## Other programming concepts / tools

### Basic coding skills**

If you've been accepted into this programme, then you probably already have this box ticked! However, polishing this area can't hurt. LeetCode is a good place to keep basic coding skills sharp, in particular practising the planning and implementation of functions in the medium-hard sections of LeetCode might be helpful. Practising problems on [Project Euler](https://projecteuler.net/) is another idea.

### VSCode**

Although many of you might already be familiar with Jupyter Notebooks, we recommend working through structured exercises using VSCode. This is a powerful text editor which provides more features than Jupyter Notebooks. Some features it has are:

- **Shortcuts**
    
    These are much more powerful than anything offered by Jupyter Notebooks. Here are a list of particularly useful ones. You can see how they all work in more detail [here](https://www.geeksforgeeks.org/visual-studio-code-shortcuts-for-windows-and-mac/).
    
    """)

    st_image("vscode-shortcuts.png", 500)

    st.markdown(r"""
    
- **Type checking**
    
    We discussed the `typing` module in a section above. This is particularly powerful when used alongside VSCode's type checker extension. You can activate typing by going to the `settings.json` file in VSCode, and adding this line:
    
    ```json
    {
        "python.analysis.typeCheckingMode": "basic"
    }
    ```
    
    You can open the `settings.json` file by first opening VSCode's Command Palette (see the shortcuts above), then finding the option **Preferences: Open User Settings (JSON)**.
    
    If you're finding that the type checker is throwing up too many warnings, you can suppress them by adding the comment `# type: ignore` at the end of a line, or using the `cast` function from the `typing` library to let the Python interpreter know what type it's dealing with. Sometimes, this is the best option. However, in general you should try and avoid doing this when you can, because type checker warnings usually mean there's a better way for you to be writing your code.
    
- **Notebook functionality**
    
    Although VSCode does provide an extension which acts just like a Jupyter Notebook, it actually has a much more useful feature. Python files can also be made to act like notebooks, by adding the line`#%%` which act as cell dividers. In this way, you can separate chunks of code and run them individually (and see their output in a new window). See [this page](https://code.visualstudio.com/docs/python/jupyter-support-py) for a further explanation.
    
- **Debugger**
    
    The VSCode debugger is a great tool, and can be much more powerful and efficient than the standard practice of adding lines to print information about your output! You can set breakpoints in your code, and closely examine the local variables which are stored at that point in the program. More information can be found on [this page](https://lightrun.com/debugging/debug-python-in-vscode/).

### Git**

Git is a piece of version control software, designed for tracking and managing changes in a set of files. It can be very powerful, but also a bit of a headache to use. Insert obligatory second xkcd:

""")

    st_image("xkcd_2.png", 300)

    st.markdown(r"""

If you already have a strong SWE background then you might not need to spend as much time on this section. Otherwise, here are a few resources you might find useful for learning the basics of Git:

- [An Intro to Git and GitHub for Beginners](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners)
- [Git Immersion](https://gitimmersion.com/index.html)
- [Git cheat sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)

Ideally, you should feel comfortable with the following:

- Cloning a repository
- Creating and switching between branches
- Staging and committing changes
- Pushing branches

### Jupyter Notebook / Colab*

Jupyter Notebooks still have some advantages over VSCode, primarily in data exploration and visualisation. At the end of each week you will be uploading your projects to GitHub, and doing this in notebook form might be a good idea.

Colab has a similar structure to Jupyter Notebooks, but it provides several additional features, most importantly GPU access. It's not yet confirmed how much we'll be using Colab and Notebooks relative to VSCode, so this section is pending updates, but it might well be worth playing around with a Colab notebook and getting the hang of how it works, e.g. how to use the built-in keyboard shortcuts to create/delete/split/run cells.

### Unix*

This won't matter much for the first 2 weeks, but it may come into play more during the “training at scale” week. [**Learning the Shell**](http://www.ee.surrey.ac.uk/Teaching/Unix/) provides a comprehensive introduction, as does **[UNIX Tutorial for Beginners](http://www.ee.surrey.ac.uk/Teaching/Unix/).**
""")

def show_double():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#maths">Maths</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#neural-networks**">Neural Networks**</a></li>
        <li><a class="contents-el" href="#linear-algebra**">Linear algebra**</a></li>
        <li><a class="contents-el" href="#probability**">Probability**</a></li>
        <li><a class="contents-el" href="#calculus**">Calculus**</a></li>
    </ul></li>
    <li><a class="contents-el" href="#programming">Programming</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#python">Python**</a></li>
        <li><a class="contents-el" href="#libraries">Libraries</a></li>
        <li><ul class="contents">
            <li><a class="contents-el" href="#numpy">numpy**</a></li>
            <li><a class="contents-el" href="#pytorch">pytorch**</a></li>
        </ul></li>
    </ul></li>
    <li><a class="contents-el" href="#other-programming-concepts-tools">Other programming concepts / tools</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#basic-coding-skills">Basic coding skills**</a></li>
        <li><a class="contents-el" href="#vscode">VSCode**</a></li>
        <li><a class="contents-el" href="#git">Git**</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""

## Maths

### Neural Networks**

We won't assume any deep knowledge of neural networks or machine learning before the programme starts, but it's useful to have an idea of the basis so that the first week doesn't have quite as steep a learning curve. The best introductory resources here are 3B1B's videos on neural networks:

- **[But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk)**
- **[Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w)**
- **[What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U)**

You should prioritise the first two videos in this sequence.

Some questions you should be able to answer after this:

""")
    st.error(r"""
- **Why do you need activation functions? Why couldn't you just create a neural network by connecting up a bunch of linear layers?**
- **What makes neural networks more powerful than basic statistical methods like linear regression?**
- **What are the advantages of ReLU activations over sigmoids?**""")

    st.markdown(r"""

### Linear algebra**

Linear algebra lies at the core of a lot of machine learning. Insert obligatory xkcd:

""")

    st_image("xkcd_1.png", 300)

    st.markdown(r"""

Here is a list of things you should probably be comfortable with:

- Linear transformations - what they are, and why they are important
    - See [this video](https://www.youtube.com/watch?v=kYB8IZa5AuE) from 3B1B""")

    st.error("""
**What is the problem in trying to create a neural network using only linear transformations?**""")

    st.markdown(r"""
- How [matrix multiplication works](http://mlwiki.org/index.php/Matrix-Matrix_Multiplication)
- Basic matrix properties: rank, trace, determinant, transpose
- Bases, and basis transformations""")

    st.error(r"""
**Read the “[Privileged vs Free Basis](https://transformer-circuits.pub/2021/framework/index.html#def-privileged-basis)” section of Anthropic's “Mathematical Framework for Transformer Circuits” paper. Which kinds of operations might lead to the creation of privileged bases? Which linear operations would preserve a privileged basis? (you might want to come back to this question once you're more familiar with the basics of neural networks and activation functions)**""")
    st.markdown(r"""
- Eigenvalues and eigenvectors
- Different types of matrix, and their significance (e.g. symmetric, orthogonal, identity, rotation matrices)

[This video series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) by 3B1B provides a good overview of these core topics (although you can probably skip it if you already have a reasonably strong mathematical background).

If you have a lot more time, [Linear Algebra Done Right](https://link.springer.com/book/10.1007/978-3-319-11080-6) is the canonical textbook for covering this topic (although it will probably cover much more than you need to know). Alternatively, Neel Nanda has two [YouTube](https://www.youtube.com/watch?v=GkPhwnvRe-8) [videos](https://www.youtube.com/watch?v=0EB23unfLSU) covering linear algebra extensively. 

### Probability**

It's essential to understand the rules of probability, expected value and standard deviation, and helpful to understand independence and the normal distribution.

""")

    st.error(r"""**What is the expected value and variance of the sum of two normally distributed random variables $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_1 \sim N(\mu_2, \sigma_2^2)$? Can you prove this mathematically?**""")

    st.markdown(r"""

### Calculus**

It's essential to understand differentiation and partial differentiation, and helpful to understand the basics of vector calculus including the chain rule and Taylor series.

Again, 3Blue1Brown has a good [video series](https://www.3blue1brown.com/topics/calculus) on this.""")

    st.error(r"""
**Read [this PyTorch documentation page](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) on the MSELoss. What is the derivative of this loss wrt the element $y_n$ (assuming we use mean reduction)? Same question for [this page](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) on BCELoss.**""")

    st.markdown(r"""

## Programming

### Python

It's important to be strong in Python, because this is the language we'll be using during the programme. As a rough indication, we expect you to be comfortable with at least 80-90% of the material  [here](https://book.pythontips.com/en/latest/), up to 21. for/else. For a more thorough treatment of Python's core functionality, see [here](https://docs.python.org/3/tutorial/).

### Libraries

The following libraries would be useful to know to at least a basic level, before the course starts.

### [`numpy`](https://numpy.org/)**

Being familiar with NumPy is a staple for working with high-performance Python. Additionally, the syntax for working with NumPy arrays is very similar to how you work with PyTorch tensors (often there are only minor differences, e.g. Torch tends to use the keyword axis where NumPy uses dim). Working through [these 100 basic NumPy exercises](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.ipynb) would be a good idea, or if you're comfortable with NumPy already then you could try doing them in PyTorch (see below).

### [`pytorch`](https://pytorch.org/)**

We will be starting week 0 of the programme with some structured exercises designed to get everyone familiar with working in PyTorch. However, the more comfortable you are with PyTorch going in, the easier you'll probably find this. PyTorch has several useful tutorials, and to get comfortable working with tensors you might want to implement the 100 basic NumPy exercises linked to above, using PyTorch instead.

## Other programming concepts / tools

### Basic coding skills**

If you've been accepted into this programme, then you probably already have this box ticked! However, polishing this area can't hurt. LeetCode is a good place to keep basic coding skills sharp, in particular practising the planning and implementation of functions in the medium-hard sections of LeetCode might be helpful. Practising problems on [Project Euler](https://projecteuler.net/) is another idea.

### VSCode**

Although many of you might already be familiar with Jupyter Notebooks, we recommend working through structured exercises using VSCode. This is a powerful text editor which provides more features than Jupyter Notebooks. Some features it has are:

- **Shortcuts**
    
    These are much more powerful than anything offered by Jupyter Notebooks. Here are a list of particularly useful ones. You can see how they all work in more detail [here](https://www.geeksforgeeks.org/visual-studio-code-shortcuts-for-windows-and-mac/).
    
    """)

    st_image("vscode-shortcuts.png", 500)

    st.markdown(r"""
    
- **Type checking**
    
    We discussed the `typing` module in a section above. This is particularly powerful when used alongside VSCode's type checker extension. You can activate typing by going to the `settings.json` file in VSCode, and adding this line:
    
    ```json
    {
        "python.analysis.typeCheckingMode": "basic"
    }
    ```
    
    You can open the `settings.json` file by first opening VSCode's Command Palette (see the shortcuts above), then finding the option **Preferences: Open User Settings (JSON)**.
    
    If you're finding that the type checker is throwing up too many warnings, you can suppress them by adding the comment `# type: ignore` at the end of a line, or using the `cast` function from the `typing` library to let the Python interpreter know what type it's dealing with. Sometimes, this is the best option. However, in general you should try and avoid doing this when you can, because type checker warnings usually mean there's a better way for you to be writing your code.
    
- **Notebook functionality**
    
    Although VSCode does provide an extension which acts just like a Jupyter Notebook, it actually has a much more useful feature. Python files can also be made to act like notebooks, by adding the line`#%%` which act as cell dividers. In this way, you can separate chunks of code and run them individually (and see their output in a new window). See [this page](https://code.visualstudio.com/docs/python/jupyter-support-py) for a further explanation.
    
- **Debugger**
    
    The VSCode debugger is a great tool, and can be much more powerful and efficient than the standard practice of adding lines to print information about your output! You can set breakpoints in your code, and closely examine the local variables which are stored at that point in the program. More information can be found on [this page](https://lightrun.com/debugging/debug-python-in-vscode/).

### Git**

Git is a piece of version control software, designed for tracking and managing changes in a set of files. It can be very powerful, but also a bit of a headache to use. Insert obligatory second xkcd:

""")

    st_image("xkcd_2.png", 300)

    st.markdown(r"""

If you already have a strong SWE background then you might not need to spend as much time on this section. Otherwise, here are a few resources you might find useful for learning the basics of Git:

- [An Intro to Git and GitHub for Beginners](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners)
- [Git Immersion](https://gitimmersion.com/index.html)
- [Git cheat sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)

Ideally, you should feel comfortable with the following:

- Cloning a repository
- Creating and switching between branches
- Staging and committing changes
- Pushing branches
""")

if slider == 0:
    with cols[1]:
        st.markdown("##### Currently showing **all content**.")
    show_all()
elif slider == 1:
    show_single_or_double()
    with cols[1]:
        st.markdown("##### Currently showing **only asterisked content**.")
elif slider == 2:
    show_double()
    with cols[1]:
        st.markdown("##### Currently showing **only double-asterisked content**.")

